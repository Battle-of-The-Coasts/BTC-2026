"""SQLite persistence for the Bluesky graph. One file, WAL mode, safe for one writer + readers.

Tables
  actors        one row per account ever seen (did is the key; handle may change)
  follows       src_did follows dst_did
  interactions  src_did -> dst_did, kind in {reply, repost, quote, mention}, n = count seen
  crawl         per-actor crawl bookkeeping: when each list was fetched, whether it was truncated by the caps
  queue         persisted BFS frontier (priority, depth, FIFO order) so a restart resumes where it stopped
  seeds         labelled accounts driving the propagation (label 1 = bot / untrusted, 0 = trusted)
  scores        latest trust score per actor + the run that produced it
  runs          one row per scoring run (parameters, sizes, convergence)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS actors (
  did TEXT PRIMARY KEY, handle TEXT, display_name TEXT, created_at TEXT, avatar TEXT,
  followers_count INTEGER, follows_count INTEGER, posts_count INTEGER, labels TEXT,
  profile_fetched_at REAL, depth INTEGER, first_seen REAL
);
CREATE INDEX IF NOT EXISTS actors_handle ON actors(handle);
CREATE TABLE IF NOT EXISTS follows (src TEXT, dst TEXT, seen_at REAL, PRIMARY KEY (src, dst)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS follows_dst ON follows(dst);
CREATE TABLE IF NOT EXISTS interactions (src TEXT, dst TEXT, kind TEXT, n INTEGER, seen_at REAL,
  PRIMARY KEY (src, dst, kind)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS interactions_dst ON interactions(dst);
CREATE TABLE IF NOT EXISTS crawl (
  did TEXT PRIMARY KEY, followers_at REAL, follows_at REAL, feed_at REAL,
  followers_truncated INTEGER DEFAULT 0, follows_truncated INTEGER DEFAULT 0, error TEXT, error_at REAL
);
CREATE TABLE IF NOT EXISTS queue (
  did TEXT PRIMARY KEY, priority INTEGER, depth INTEGER, seq INTEGER, enqueued_at REAL
);
CREATE INDEX IF NOT EXISTS queue_order ON queue(priority, depth, seq);
CREATE TABLE IF NOT EXISTS seeds (did TEXT PRIMARY KEY, label INTEGER, source TEXT, added_at REAL);
CREATE TABLE IF NOT EXISTS scores (did TEXT PRIMARY KEY, trust REAL, residual REAL, run_id INTEGER);
CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, computed_at REAL, n_nodes INTEGER,
  n_edges INTEGER, n_seeds INTEGER, params TEXT, iterations INTEGER, converged INTEGER);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(actors)")}
        if "verified" not in cols:      # migration: 0 = not verified, 1 = verified, 2 = trusted verifier, NULL = unknown
            self.db.execute("ALTER TABLE actors ADD COLUMN verified INTEGER")
        self._seq = int(self.db.execute("SELECT COALESCE(MAX(seq), 0) FROM queue").fetchone()[0])

    # ---- generic ------------------------------------------------------------------------------
    def tx(self):
        return _Tx(self)

    def q(self, sql: str, *args) -> list[sqlite3.Row]:
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def one(self, sql: str, *args):
        with self.lock:
            return self.db.execute(sql, args).fetchone()

    def counts(self) -> dict:
        return {
            "actors": self.one("SELECT COUNT(*) FROM actors")[0],
            "actors_with_profile": self.one("SELECT COUNT(*) FROM actors WHERE profile_fetched_at IS NOT NULL")[0],
            "crawled": self.one("SELECT COUNT(*) FROM crawl WHERE followers_at IS NOT NULL")[0],
            "follows": self.one("SELECT COUNT(*) FROM follows")[0],
            "interactions": self.one("SELECT COALESCE(SUM(n),0) FROM interactions")[0],
            "queue": self.one("SELECT COUNT(*) FROM queue")[0],
            "seeds": self.one("SELECT COUNT(*) FROM seeds")[0],
            "scored": self.one("SELECT COUNT(*) FROM scores")[0],
        }

    # ---- actors -------------------------------------------------------------------------------
    def upsert_actor(self, a: dict, depth: int | None = None, full_profile: bool = False):
        """a: an AppView profile / profileViewBasic dict."""
        did = a.get("did")
        if not did:
            return
        now = time.time()
        labels = json.dumps([l.get("val") for l in a.get("labels") or [] if l.get("val")]) if "labels" in a else None
        ver = a.get("verification")
        if isinstance(ver, dict):
            verified = 2 if ver.get("trustedVerifierStatus") == "valid" else 1 if ver.get("verifiedStatus") == "valid" else 0
        else:
            verified = 0 if full_profile else None     # listings only carry the field for verified accounts
        with self.lock:
            self.db.execute(
                """INSERT INTO actors (did, handle, display_name, created_at, avatar, followers_count, follows_count,
                       posts_count, labels, profile_fetched_at, depth, first_seen, verified)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(did) DO UPDATE SET
                       handle=COALESCE(excluded.handle, actors.handle),
                       display_name=COALESCE(excluded.display_name, actors.display_name),
                       created_at=CASE WHEN excluded.profile_fetched_at IS NULL AND actors.profile_fetched_at IS NOT NULL
                                       THEN actors.created_at ELSE COALESCE(excluded.created_at, actors.created_at) END,
                       avatar=COALESCE(excluded.avatar, actors.avatar),
                       followers_count=CASE WHEN excluded.profile_fetched_at IS NULL AND actors.profile_fetched_at IS NOT NULL
                                       THEN actors.followers_count ELSE COALESCE(excluded.followers_count, actors.followers_count) END,
                       follows_count=CASE WHEN excluded.profile_fetched_at IS NULL AND actors.profile_fetched_at IS NOT NULL
                                       THEN actors.follows_count ELSE COALESCE(excluded.follows_count, actors.follows_count) END,
                       posts_count=CASE WHEN excluded.profile_fetched_at IS NULL AND actors.profile_fetched_at IS NOT NULL
                                       THEN actors.posts_count ELSE COALESCE(excluded.posts_count, actors.posts_count) END,
                       labels=COALESCE(excluded.labels, actors.labels),
                       profile_fetched_at=COALESCE(excluded.profile_fetched_at, actors.profile_fetched_at),
                       depth=CASE WHEN excluded.depth IS NULL THEN actors.depth
                                  WHEN actors.depth IS NULL THEN excluded.depth ELSE MIN(actors.depth, excluded.depth) END,
                       verified=COALESCE(excluded.verified, actors.verified)""",
                (did, a.get("handle"), a.get("displayName"), a.get("createdAt"), a.get("avatar"),
                 a.get("followersCount"), a.get("followsCount"), a.get("postsCount"), labels,
                 now if full_profile else None, depth, now, verified))

    def actor(self, did: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM actors WHERE did=?", did)

    def resolve(self, ident: str) -> sqlite3.Row | None:
        """ident = handle (case-insensitive) or did."""
        if ident.startswith("did:"):
            return self.actor(ident)
        return self.one("SELECT * FROM actors WHERE handle=? COLLATE NOCASE ORDER BY profile_fetched_at DESC LIMIT 1", ident)

    # ---- edges --------------------------------------------------------------------------------
    def add_follows(self, pairs: Iterable[tuple[str, str]]):
        now = time.time()
        with self.lock:
            self.db.executemany("INSERT OR IGNORE INTO follows (src, dst, seen_at) VALUES (?,?,?)",
                                [(s, d, now) for s, d in pairs if s != d])

    def add_interactions(self, triples: Iterable[tuple[str, str, str]]):
        now = time.time()
        agg: dict[tuple[str, str, str], int] = {}
        for s, d, k in triples:
            if s != d:
                agg[(s, d, k)] = agg.get((s, d, k), 0) + 1
        with self.lock:
            self.db.executemany(
                """INSERT INTO interactions (src, dst, kind, n, seen_at) VALUES (?,?,?,?,?)
                   ON CONFLICT(src, dst, kind) DO UPDATE SET n=MAX(interactions.n, excluded.n), seen_at=excluded.seen_at""",
                [(s, d, k, n, now) for (s, d, k), n in agg.items()])

    def replace_interactions_from(self, src: str, triples: list[tuple[str, str, str]]):
        """The author feed is a snapshot; re-crawling replaces what we knew from that actor."""
        with self.lock:
            self.db.execute("DELETE FROM interactions WHERE src=?", (src,))
            self.add_interactions(triples)

    def neighbours(self, did: str) -> dict[str, set[str]]:
        f_out = {r[0] for r in self.q("SELECT dst FROM follows WHERE src=?", did)}
        f_in = {r[0] for r in self.q("SELECT src FROM follows WHERE dst=?", did)}
        i_out = {r[0] for r in self.q("SELECT dst FROM interactions WHERE src=?", did)}
        i_in = {r[0] for r in self.q("SELECT src FROM interactions WHERE dst=?", did)}
        return {"follows": f_out, "followers": f_in, "interacts_with": i_out, "interacted_by": i_in}

    # ---- crawl state --------------------------------------------------------------------------
    def crawl_state(self, did: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM crawl WHERE did=?", did)

    def mark_crawled(self, did: str, followers_truncated: bool, follows_truncated: bool, error: str | None = None):
        now = time.time()
        with self.lock:
            if error:
                self.db.execute("""INSERT INTO crawl (did, error, error_at) VALUES (?,?,?)
                                   ON CONFLICT(did) DO UPDATE SET error=excluded.error, error_at=excluded.error_at""",
                                (did, error, now))
            else:
                self.db.execute("""INSERT INTO crawl (did, followers_at, follows_at, feed_at, followers_truncated,
                                       follows_truncated, error, error_at) VALUES (?,?,?,?,?,?,NULL,NULL)
                                   ON CONFLICT(did) DO UPDATE SET followers_at=excluded.followers_at,
                                       follows_at=excluded.follows_at, feed_at=excluded.feed_at,
                                       followers_truncated=excluded.followers_truncated,
                                       follows_truncated=excluded.follows_truncated, error=NULL, error_at=NULL""",
                                (did, now, now, now, int(followers_truncated), int(follows_truncated)))

    def is_fresh(self, did: str, ttl_seconds: float) -> bool:
        c = self.crawl_state(did)
        return bool(c and c["followers_at"] and time.time() - c["followers_at"] < ttl_seconds)

    # ---- BFS queue ----------------------------------------------------------------------------
    def enqueue(self, did: str, priority: int, depth: int) -> bool:
        """Insert or upgrade (lower priority number / lower depth wins; FIFO within the same class)."""
        with self.lock:
            row = self.db.execute("SELECT priority, depth FROM queue WHERE did=?", (did,)).fetchone()
            if row is None:
                self._seq += 1
                self.db.execute("INSERT INTO queue (did, priority, depth, seq, enqueued_at) VALUES (?,?,?,?,?)",
                                (did, priority, depth, self._seq, time.time()))
                return True
            if (priority, depth) < (row["priority"], row["depth"]):
                self.db.execute("UPDATE queue SET priority=?, depth=? WHERE did=?", (priority, depth, did))
                return True
            return False

    def dequeue(self) -> sqlite3.Row | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM queue ORDER BY priority, depth, seq LIMIT 1").fetchone()
            if row is not None:
                self.db.execute("DELETE FROM queue WHERE did=?", (row["did"],))
            return row

    def queued(self, did: str) -> bool:
        return self.one("SELECT 1 FROM queue WHERE did=?", did) is not None

    def queue_summary(self) -> dict:
        rows = self.q("SELECT priority, depth, COUNT(*) c FROM queue GROUP BY priority, depth ORDER BY priority, depth")
        return {"total": sum(r["c"] for r in rows), "by_class": [dict(r) for r in rows]}

    # ---- seeds --------------------------------------------------------------------------------
    def set_seed(self, did: str, label: int, source: str, keep_manual: bool = False, keep: tuple = ("manual",)):
        """keep_manual: do not overwrite a seed whose source is in `keep` (by default one the user set by hand)."""
        with self.lock:
            if keep_manual and self.db.execute(f"SELECT 1 FROM seeds WHERE did=? AND source IN ({','.join('?' * len(keep))})",
                                               (did, *keep)).fetchone():
                return
            self.db.execute("INSERT OR REPLACE INTO seeds (did, label, source, added_at) VALUES (?,?,?,?)",
                            (did, int(label), source, time.time()))

    def remove_seed(self, did: str):
        with self.lock:
            self.db.execute("DELETE FROM seeds WHERE did=?", (did,))

    def clear_seeds(self, source: str | None = None):
        with self.lock:
            if source:
                self.db.execute("DELETE FROM seeds WHERE source=?", (source,))
            else:
                self.db.execute("DELETE FROM seeds")

    def seeds(self) -> list[sqlite3.Row]:
        return self.q("""SELECT s.did, s.label, s.source, s.added_at, a.handle, a.display_name, sc.trust
                         FROM seeds s LEFT JOIN actors a ON a.did=s.did LEFT JOIN scores sc ON sc.did=s.did
                         ORDER BY s.added_at""")

    # ---- scores -------------------------------------------------------------------------------
    def write_scores(self, rows: list[tuple[str, float, float]], params: dict, n_nodes: int, n_edges: int,
                     n_seeds: int, iterations: int, converged: bool) -> int:
        with self.lock:
            cur = self.db.execute("INSERT INTO runs (computed_at, n_nodes, n_edges, n_seeds, params, iterations, converged) "
                                  "VALUES (?,?,?,?,?,?,?)",
                                  (time.time(), n_nodes, n_edges, n_seeds, json.dumps(params), iterations, int(converged)))
            run_id = int(cur.lastrowid)
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM scores")
            self.db.executemany("INSERT INTO scores (did, trust, residual, run_id) VALUES (?,?,?,?)",
                                [(d, t, r, run_id) for d, t, r in rows])
            self.db.execute("COMMIT")
            return run_id

    def last_run(self) -> dict | None:
        r = self.one("SELECT * FROM runs ORDER BY run_id DESC LIMIT 1")
        if r is None:
            return None
        d = dict(r)
        d["params"] = json.loads(d["params"])
        return d

    def score(self, did: str) -> sqlite3.Row | None:
        return self.one("SELECT * FROM scores WHERE did=?", did)

    # ---- settings -----------------------------------------------------------------------------
    def get_setting(self, key: str, default):
        r = self.one("SELECT value FROM settings WHERE key=?", key)
        return json.loads(r[0]) if r else default

    def set_setting(self, key: str, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, json.dumps(value)))


class _Tx:
    def __init__(self, store: Store):
        self.s = store

    def __enter__(self):
        self.s.lock.acquire()
        self.s.db.execute("BEGIN")
        return self.s

    def __exit__(self, et, ev, tb):
        try:
            self.s.db.execute("ROLLBACK" if et else "COMMIT")
        finally:
            self.s.lock.release()
