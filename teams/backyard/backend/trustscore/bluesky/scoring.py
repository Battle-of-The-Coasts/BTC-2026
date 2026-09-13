"""Trust propagation on the crawled Bluesky graph, reusing the step-1 formula unchanged:

    r = (1 - alpha) q + alpha P r ,   trust = 1/2 - r
    q = +1/2 for bot seeds, -1/2 for trusted seeds, lam * (p_local - 1/2) otherwise, with p_local the transparent
        profile prior of local_prior.py (account age, handle pattern, avatar, counts, moderation labels)
    P = row-normalised, hub-damped, homophily-weighted gather matrix over the channels
        mutual / follow_in / follow_out / reply_in|out / repost_in|out / quote_in|out / mention_in|out

Seeds: whatever is in the `seeds` table, by source:
  manual        set from the popover / dashboard (always wins)
  list          curated news outlets + verified accounts found by search (trusted_lists.py)
  random        optional random draw of crawled accounts, labelled trusted (the first placeholder)
  rule:verified      trusted: the AppView reports the account as verified (blue check / trusted verifier)
  labeler            untrusted: moderation label spam / bot / impersonation / ...
  rule:new_account   untrusted: created less than `rule_new_account_days` ago
  rule:few_followers untrusted: fewer than `rule_min_followers` followers (only when the count is known)
Rule seeds are re-derived from the actors table before every run (in that precedence order); the others persist.
"""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import numpy as np
import scipy.sparse as sp

from ..config import PropagationConfig
from ..datasets import GraphData
from ..graph import build_channels, combine_channels
from ..propagation import build_prior, explain_node, propagate, trust_from_residual
from .local_prior import BOT_LABELS, profile_evidence
from .store import Store


@dataclass
class ScoringSettings:
    alpha: float = 0.8
    seed_residual: float = 0.5
    hub_damping: float = 0.5
    random_seed_count: int = 50
    auto_label_seeds: bool = True        # moderation labels -> untrusted seeds
    rule_verified_trusted: bool = True   # verified accounts -> trusted seeds
    rule_bots: bool = True               # age / follower rules -> untrusted seeds
    rule_new_account_days: float = 7.0
    rule_min_followers: int = 20
    min_edges_for_seed: int = 5      # random seeds are drawn among crawled accounts with at least this many edges
    local_prior: bool = True         # q = lam * (p_local - 1/2) for non-seed nodes (local_prior.py)
    lam: float = 0.5                 # confidence given to the profile prior (step-1 value)
    min_neighbours_for_score: int = 3   # uncrawled accounts with fewer stored edges are reported as "not enough graph"
    channel_weights: dict = field(default_factory=lambda: {
        "mutual": 1.0, "follow_in": 0.5, "follow_out": 0.25,
        "reply_in": 0.5, "reply_out": 0.5, "repost_in": 0.5, "repost_out": 0.5,
        "quote_in": 0.5, "quote_out": 0.5, "mention_in": 0.5, "mention_out": 0.5,
    })

    @classmethod
    def load(cls, store: Store) -> "ScoringSettings":
        d = store.get_setting("scoring", {})
        s = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "channel_weights"})
        if "channel_weights" in d:
            s.channel_weights.update(d["channel_weights"])
        return s

    def save(self, store: Store):
        store.set_setting("scoring", asdict(self))


class Graph:
    """Snapshot of the store as sparse matrices (node index = position in `dids`)."""

    def __init__(self, store: Store):
        rows = store.q("SELECT did FROM actors ORDER BY rowid")
        self.dids = [r[0] for r in rows]
        self.idx = {d: i for i, d in enumerate(self.dids)}
        n = len(self.dids)
        rel: dict[str, sp.csr_matrix] = {}
        f = store.q("SELECT src, dst FROM follows")
        rel["follow"] = self._mat(n, [(s, d, 1.0) for s, d in f])
        for kind in ("reply", "repost", "quote", "mention"):
            e = store.q("SELECT src, dst, n FROM interactions WHERE kind=?", kind)
            rel[kind] = self._mat(n, [(s, d, float(w)) for s, d, w in e])
        self.gd = GraphData(name="bluesky", node_ids=np.arange(n), y=np.full(n, -1, dtype=np.int8),
                            subset=np.array([""] * n), relations=rel)
        self.n = n
        self.n_edges = int(sum(int(m.nnz) for m in rel.values()))

    def _mat(self, n: int, triples) -> sp.csr_matrix:
        r, c, v = [], [], []
        for s, d, w in triples:
            i, j = self.idx.get(s), self.idx.get(d)
            if i is not None and j is not None and i != j:
                r.append(i); c.append(j); v.append(w)
        # convention of graph.build_channels: A[u, v] = u -> v (u follows v / u acts on v)
        return sp.csr_matrix((np.asarray(v, dtype=np.float64), (r, c)), shape=(n, n))


class Scorer:
    def __init__(self, store: Store, settings: ScoringSettings | None = None):
        self.store = store
        self.settings = settings or ScoringSettings.load(store)
        self.graph: Graph | None = None
        self.channels: dict[str, sp.csr_matrix] = {}
        self.q: np.ndarray | None = None
        self.r: np.ndarray | None = None
        self.trust: np.ndarray | None = None
        self.computed_at: float | None = None
        self.info: dict = {}

    # ---- seeds ----------------------------------------------------------------------------------
    def draw_random_seeds(self, k: int | None = None, rng_seed: int | None = None) -> list[str]:
        k = k or self.settings.random_seed_count
        rows = self.store.q(
            """SELECT a.did, (SELECT COUNT(*) FROM follows f WHERE f.src=a.did OR f.dst=a.did) AS deg
               FROM actors a JOIN crawl c ON c.did=a.did WHERE c.followers_at IS NOT NULL""")
        pool = [r["did"] for r in rows if r["deg"] >= self.settings.min_edges_for_seed] or [r["did"] for r in rows]
        rng = random.Random(rng_seed if rng_seed is not None else time.time_ns())
        chosen = rng.sample(pool, min(k, len(pool)))
        self.store.clear_seeds(source="random")
        for d in chosen:
            self.store.set_seed(d, 0, "random")
        return chosen

    def apply_seed_rules(self) -> dict:
        """Re-derive rule seeds from the actors table. INSERT OR IGNORE keeps manual / list / random seeds and gives
        earlier rules precedence (verified beats the bot rules)."""
        s = self.settings
        now = time.time()
        counts = {}
        with self.store.lock:
            db = self.store.db
            db.execute("DELETE FROM seeds WHERE source LIKE 'rule:%' OR source='labeler'")
            if s.rule_verified_trusted:
                counts["rule:verified"] = db.execute(
                    "INSERT OR IGNORE INTO seeds (did, label, source, added_at) SELECT did, 0, 'rule:verified', ? FROM actors "
                    "WHERE verified >= 1", (now,)).rowcount
            if s.auto_label_seeds:
                like = " OR ".join("labels LIKE ?" for _ in BOT_LABELS)
                counts["labeler"] = db.execute(
                    f"INSERT OR IGNORE INTO seeds (did, label, source, added_at) SELECT did, 1, 'labeler', ? FROM actors "
                    f"WHERE labels IS NOT NULL AND ({like})", (now, *[f'%"{l}"%' for l in BOT_LABELS])).rowcount
            if s.rule_bots:
                cutoff = datetime.fromtimestamp(now - s.rule_new_account_days * 86400, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                counts["rule:new_account"] = db.execute(
                    "INSERT OR IGNORE INTO seeds (did, label, source, added_at) SELECT did, 1, 'rule:new_account', ? FROM actors "
                    "WHERE created_at > ? AND created_at <= ?", (now, cutoff, "9")).rowcount
                counts["rule:few_followers"] = db.execute(
                    "INSERT OR IGNORE INTO seeds (did, label, source, added_at) SELECT did, 1, 'rule:few_followers', ? FROM actors "
                    "WHERE followers_count IS NOT NULL AND followers_count < ?", (now, int(s.rule_min_followers))).rowcount
        return counts

    # ---- run ------------------------------------------------------------------------------------
    def run(self) -> dict:
        s = self.settings
        rule_counts = self.apply_seed_rules()
        g = Graph(self.store)
        if g.n == 0:
            return {"n_nodes": 0}
        seeds = self.store.q("SELECT did, label FROM seeds")
        sidx = np.array([g.idx[r["did"]] for r in seeds if r["did"] in g.idx], dtype=np.int64)
        sy = np.array([r["label"] for r in seeds if r["did"] in g.idx], dtype=np.int8)
        p_local = None
        if s.local_prior:
            p_local = np.full(g.n, 0.5)
            for row in self.store.q("SELECT did, handle, display_name, created_at, avatar, followers_count, follows_count, "
                                    "posts_count, labels FROM actors"):
                i = g.idx.get(row["did"])
                if i is not None:
                    p_local[i] = profile_evidence(row)[0]
        q = build_prior(g.n, sidx, sy, p_local, lam=s.lam, seed_residual=s.seed_residual)
        channels = build_channels(g.gd, hub_damping=s.hub_damping)
        cfg = PropagationConfig(alpha=s.alpha, seed_residual=s.seed_residual, hub_damping=s.hub_damping)
        try:
            P = combine_channels(channels, s.channel_weights)
            r, info = propagate(P, q, alpha=s.alpha, max_iter=cfg.max_iter, tol=cfg.tol, return_info=True)
        except ValueError:      # no edge at all yet
            r, info = q.copy(), {"iterations": 0, "converged": True}
        trust = trust_from_residual(r)
        self.graph, self.channels, self.q, self.r, self.trust = g, channels, q, r, trust
        self.computed_at = time.time()
        params = {"alpha": s.alpha, "seed_residual": s.seed_residual, "hub_damping": s.hub_damping, "lam": s.lam if s.local_prior else 0.0,
                  "channel_weights": s.channel_weights, "n_seeds_bot": int((sy == 1).sum()), "n_seeds_trusted": int((sy == 0).sum())}
        run_id = self.store.write_scores([(g.dids[i], float(trust[i]), float(r[i])) for i in range(g.n)], params,
                                         g.n, g.n_edges, len(sidx), info["iterations"], info["converged"])
        self.info = {"run_id": run_id, "n_nodes": g.n, "n_edges": g.n_edges, "n_seeds": int(len(sidx)), **info,
                     "computed_at": self.computed_at, "params": params, "rule_seeds": rule_counts}
        return self.info

    # ---- explanation ----------------------------------------------------------------------------
    def explain(self, did: str, top_k: int = 8) -> dict | None:
        if self.graph is None or did not in self.graph.idx:
            return None
        v = self.graph.idx[did]
        ex = explain_node(v, self.channels, self.settings.channel_weights, self.q, self.r, self.settings.alpha, top_k=top_k)
        # Effect on the *trust* scale (trust = 1/2 - r): a positive residual contribution lowers trust.
        seed = self.store.one("SELECT label FROM seeds WHERE did=?", did)
        comps = profile_evidence(self.store.actor(did))[1] if self.settings.local_prior else []
        why = [["Own evidence (seed)" if seed else "Own evidence (profile prior)", -ex["prior_term"], _prior_note(self.q[v], seed, comps)]]
        for name, c in sorted(ex["per_channel"].items(), key=lambda kv: -abs(kv[1]["contribution"])):
            if c["n_neighbors"] == 0:
                continue
            why.append([CHANNEL_LABEL.get(name, name), -c["contribution"],
                        f"{c['n_neighbors']} account(s) · weight {c['homophily_weight']:.2f}"])
        nb = []
        for t in ex["top_neighbors"]:
            u = self.graph.dids[t["node"]]
            a = self.store.actor(u)
            nb.append([a["handle"] if a and a["handle"] else u[:24], CHANNEL_SHORT.get(t["channel"], t["channel"]),
                       float(self.trust[t["node"]]), -t["contribution"], u])
        return {"trust": float(self.trust[v]), "residual": float(self.r[v]), "why": why, "neighbours": nb,
                "n_neighbours": ex["n_neighbors"], "computed_at": self.computed_at,
                "prior_components": [[n, val] for n, val in comps]}


CHANNEL_LABEL = {
    "mutual": "Mutual follows", "follow_in": "Followed by", "follow_out": "Follows",
    "reply_in": "Replied to by", "reply_out": "Replies to", "repost_in": "Reposted by", "repost_out": "Reposts",
    "quote_in": "Quoted by", "quote_out": "Quotes", "mention_in": "Mentioned by", "mention_out": "Mentions",
}
CHANNEL_SHORT = {
    "mutual": "mutual", "follow_in": "follows them", "follow_out": "followed by them",
    "reply_in": "replies to them", "reply_out": "they reply to", "repost_in": "reposts them", "repost_out": "they repost",
    "quote_in": "quotes them", "quote_out": "they quote", "mention_in": "mentions them", "mention_out": "they mention",
}


def _prior_note(qv: float, seed, comps) -> str:
    if seed is not None:
        return "seed: labelled " + ("untrusted" if seed["label"] == 1 else "trusted")
    if not comps:
        return "profile prior: nothing known, p = 0.50"
    p = 0.5 + 0.5 * float(np.tanh(sum(v for _, v in comps)))
    return "profile prior p = %.2f: " % p + " · ".join(f"{n} {v:+.2f}" for n, v in comps)
