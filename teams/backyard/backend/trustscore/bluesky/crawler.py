"""Breadth-first crawler.

The add-on sends the accounts visible on the page (priority 0) and the ones just below the fold (priority 1).
Each dequeued account gets its profile, its followers (up to `list_pages` x 100), its follows (same cap) and its
last `feed_limit` posts, from which reply / repost / quote / mention edges are extracted. Every neighbour becomes a
node; the frontier is a persisted FIFO ordered by (priority, depth, arrival), so the crawl is breadth-first:
all depth-d accounts of a request are fully expanded before any depth-(d+1) account is touched.

Expansion policy (all in Settings, editable from the add-on):
  max_depth        : accounts at depth < max_depth get their neighbours enqueued at depth+1 (priority 2 = background;
                     the queue is ordered by depth inside that priority, so the whole depth-1 layer of every page
                     account is crawled before any depth-2 account)
  expand_per_node  : neighbours enqueued per page account (depth 0), mutual follows first, then interaction
                     partners, then one-way follows (bounded cost: each expansion is ~1 + 2*list_pages + 1 calls)
  expand_deep      : neighbours enqueued per deeper node (depth >= 1)
  ttl_hours        : a crawled account is not re-crawled before this delay
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass

from .client import BskyClient, BskyError, extract_interactions
from .store import Store

log = logging.getLogger("bluesky.crawler")


@dataclass
class Settings:
    max_depth: int = 2            # 0 = only the accounts on the page; 1 = their neighbours; 2 = neighbours of those
    expand_per_node: int = 25     # neighbours enqueued per page account (depth 0 -> 1), mutuals first
    expand_deep: int = 6          # neighbours enqueued per deeper node (depth >= 1), so depth 2 costs 25 x 6 per page account
    list_pages: int = 3           # pages of 100 for followers and for follows
    feed_limit: int = 100         # posts read for interactions
    ttl_hours: float = 72.0
    requests_per_second: float = 4.0
    rescore_every_seconds: float = 20.0   # re-run the propagation at most this often while crawling

    @classmethod
    def load(cls, store: Store) -> "Settings":
        return cls(**{k: v for k, v in store.get_setting("crawler", {}).items() if k in cls.__dataclass_fields__})

    def save(self, store: Store):
        store.set_setting("crawler", asdict(self))


class Crawler:
    def __init__(self, store: Store, client: BskyClient, settings: Settings | None = None, on_change=None):
        self.store = store
        self.client = client
        self.settings = settings or Settings.load(store)
        self.on_change = on_change          # callback(n_nodes_touched) after each crawled account
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.thread: threading.Thread | None = None
        self.current: str | None = None
        self.processed = 0
        self.errors = 0
        self.last_error: str | None = None

    # ---- public --------------------------------------------------------------------------------
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, name="bsky-crawler", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 5.0):
        self._stop.set(); self._wake.set()
        if self.thread:
            self.thread.join(timeout)

    def request(self, idents: list[str], priority: int = 0) -> dict:
        """Called by the API: resolve identifiers to DIDs (batched profile call for unknown ones), enqueue those
        that are stale or never crawled. Returns {ident: did|None}."""
        out: dict[str, str | None] = {}
        unknown = []
        for ident in idents:
            row = self.store.resolve(ident)
            if row is None:
                unknown.append(ident)
            else:
                out[ident] = row["did"]
        if unknown:
            try:
                for p in self.client.get_profiles(unknown):
                    self.store.upsert_actor(p, depth=0, full_profile=True)
                    for ident in unknown:
                        if ident == p.get("did") or ident.lower() == (p.get("handle") or "").lower():
                            out[ident] = p["did"]
            except BskyError as e:
                log.warning("getProfiles failed: %s", e)
            for ident in unknown:
                out.setdefault(ident, None)
        queued = []
        ttl = self.settings.ttl_hours * 3600
        for ident, did in out.items():
            if did and not self.store.is_fresh(did, ttl):
                if self.store.enqueue(did, priority, 0):
                    queued.append(did)
        if queued:
            self._wake.set()
        return out

    def status(self) -> dict:
        return {"running": bool(self.thread and self.thread.is_alive()), "current": self.current,
                "processed": self.processed, "errors": self.errors, "last_error": self.last_error,
                "api_calls": self.client.calls, "queue": self.store.queue_summary(), "settings": asdict(self.settings)}

    # ---- worker --------------------------------------------------------------------------------
    def _loop(self):
        while not self._stop.is_set():
            row = self.store.dequeue()
            if row is None:
                self._wake.wait(timeout=2.0); self._wake.clear()
                continue
            did, depth = row["did"], int(row["depth"])
            if self.store.is_fresh(did, self.settings.ttl_hours * 3600):
                continue
            self.current = did
            try:
                self.crawl_one(did, depth)
                self.processed += 1
            except BskyError as e:
                self.errors += 1; self.last_error = f"{did}: {e}"
                self.store.mark_crawled(did, False, False, error=str(e))
                log.warning("crawl %s failed: %s", did, e)
            except Exception as e:   # keep the worker alive whatever happens
                self.errors += 1; self.last_error = f"{did}: {e!r}"
                self.store.mark_crawled(did, False, False, error=repr(e))
                log.exception("crawl %s crashed", did)
            finally:
                self.current = None
            if self.on_change:
                try:
                    self.on_change()
                except Exception:
                    log.exception("on_change failed")

    def crawl_one(self, did: str, depth: int):
        s = self.settings
        prof = self.client.get_profile(did)
        self.store.upsert_actor(prof, depth=depth, full_profile=True)
        did = prof["did"]

        followers, f_trunc = self.client.followers(did, s.list_pages)
        follows, g_trunc = self.client.follows(did, s.list_pages)
        feed = self.client.author_feed(did, s.feed_limit) if s.feed_limit > 0 else []
        edges, seen = extract_interactions(did, feed)

        with self.store.tx() as st:
            for a in followers:
                st.upsert_actor(a, depth=depth + 1)
            for a in follows:
                st.upsert_actor(a, depth=depth + 1)
            for a in seen.values():
                st.upsert_actor(a, depth=depth + 1)
            for _, dst, _ in edges:
                if st.actor(dst) is None:
                    st.upsert_actor({"did": dst}, depth=depth + 1)
            st.add_follows((a["did"], did) for a in followers)
            st.add_follows((did, a["did"]) for a in follows)
            st.replace_interactions_from(did, edges)
            st.mark_crawled(did, f_trunc, g_trunc)

        if depth < s.max_depth and s.expand_per_node > 0:
            self._expand(did, depth, followers, follows, edges)

    def _expand(self, did: str, depth: int, followers: list[dict], follows: list[dict], edges):
        """BFS frontier: neighbours go in at depth+1 with background priority, mutual follows first."""
        fin = {a["did"] for a in followers}
        fout = {a["did"] for a in follows}
        order: list[str] = []
        seen: set[str] = set()

        def push(x):
            if x != did and x not in seen:
                seen.add(x); order.append(x)

        # deterministic: keep the order the AppView returned (most recent follows first)
        for a in follows:
            if a["did"] in fin:
                push(a["did"])
        for _, dst, _ in edges:
            push(dst)
        for a in followers:
            push(a["did"])
        for a in follows:
            push(a["did"])
        ttl = self.settings.ttl_hours * 3600
        cap = self.settings.expand_per_node if depth == 0 else self.settings.expand_deep
        n = 0
        for x in order:
            if n >= cap:
                break
            if self.store.is_fresh(x, ttl):
                continue
            self.store.enqueue(x, 2, depth + 1)
            n += 1
