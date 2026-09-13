"""Bluesky crawler/store/scoring tests with an in-memory fake AppView (no network)."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["BLUESKY_AUTOAPP"] = "0"

from trustscore.bluesky.client import extract_interactions   # noqa: E402
from trustscore.bluesky.crawler import Crawler, Settings       # noqa: E402
from trustscore.bluesky.scoring import Scorer, ScoringSettings  # noqa: E402
from trustscore.bluesky.store import Store                      # noqa: E402


def prof(i, labels=(), verified=False, created="2024-01-01T00:00:00Z", followers=100):
    p = {"did": f"did:plc:{i}", "handle": f"u{i}.test", "displayName": f"User {i}", "createdAt": created,
         "followersCount": followers, "followsCount": 1, "postsCount": 1, "labels": [{"val": l} for l in labels]}
    if verified:
        p["verification"] = {"verifiedStatus": "valid", "trustedVerifierStatus": "none"}
    return p


class FakeClient:
    """Graph: 0..4 a clique of mutual follows (trusted side); 5..9 a clique (bot side); 4 -> 5 one bridge follow.
    Node 2 replies to 3, node 7 reposts 8."""

    def __init__(self):
        self.calls = 0
        self.crawled_order = []
        self.follow = {i: set() for i in range(10)}
        for grp in (range(0, 5), range(5, 10)):
            for a in grp:
                for b in grp:
                    if a != b:
                        self.follow[a].add(b)
        self.follow[4].add(5)
        self.labels = {9: ["spam"]}
        self.limiter = type("L", (), {"rate": 1.0})()
        self.verified = {0}
        self.created = {6: "2099-01-01T00:00:00Z"}     # "created in the future" = certainly younger than a week
        self.follower_counts = {7: 3}

    def _prof(self, i):
        import datetime as _dt
        created = self.created.get(i)
        if created == "2099-01-01T00:00:00Z":
            created = (_dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return prof(i, self.labels.get(i, ()), i in self.verified, created or "2024-01-01T00:00:00Z", self.follower_counts.get(i, 100))

    def get_profiles(self, actors):
        self.calls += 1
        out = []
        for a in actors:
            i = int(a.split(":")[-1]) if a.startswith("did:") else int(a[1:].split(".")[0])
            out.append(self._prof(i))
        return out

    def get_profile(self, actor):
        self.calls += 1
        i = int(actor.split(":")[-1])
        self.crawled_order.append(i)
        return self._prof(i)

    def followers(self, actor, max_pages=3):
        i = int(actor.split(":")[-1])
        return [self._prof(j) for j in range(10) if i in self.follow[j]], False

    def follows(self, actor, max_pages=3):
        i = int(actor.split(":")[-1])
        return [self._prof(j) for j in sorted(self.follow[i])], False

    def author_feed(self, actor, limit=100):
        i = int(actor.split(":")[-1])
        if i == 2:
            return [{"post": {"uri": "at://did:plc:2/app.bsky.feed.post/x", "author": prof(2),
                              "record": {"reply": {"parent": {"uri": "at://did:plc:3/app.bsky.feed.post/y"}}, "text": "hi"}},
                     "reply": {"parent": {"author": prof(3)}}}]
        if i == 7:
            return [{"post": {"uri": "at://did:plc:8/app.bsky.feed.post/z", "author": prof(8), "record": {"text": "x"}},
                     "reason": {"$type": "app.bsky.feed.defs#reasonRepost", "by": prof(7)}}]
        return []


@pytest.fixture
def env():
    d = tempfile.mkdtemp()
    store = Store(os.path.join(d, "t.sqlite"))
    client = FakeClient()
    crawler = Crawler(store, client, Settings(max_depth=1, expand_per_node=3, ttl_hours=1))
    return store, client, crawler


def drain(crawler):
    while True:
        row = crawler.store.dequeue()
        if row is None:
            return
        crawler.crawl_one(row["did"], int(row["depth"]))


def test_extract_interactions_kinds():
    c = FakeClient()
    e, seen = extract_interactions("did:plc:2", c.author_feed("did:plc:2"))
    assert e == [("did:plc:2", "did:plc:3", "reply")] and "did:plc:3" in seen
    e, _ = extract_interactions("did:plc:7", c.author_feed("did:plc:7"))
    assert e == [("did:plc:7", "did:plc:8", "repost")]


def test_request_resolves_and_enqueues(env):
    store, client, crawler = env
    res = crawler.request(["u0.test", "did:plc:5"], priority=0)
    assert res == {"u0.test": "did:plc:0", "did:plc:5": "did:plc:5"}
    assert store.queue_summary()["total"] == 2
    # a second request for the same accounts does not duplicate them
    crawler.request(["u0.test"], priority=1)
    assert store.queue_summary()["total"] == 2


def test_bfs_order_and_persistence(env):
    store, client, crawler = env
    crawler.request(["u0.test"], priority=0)
    drain(crawler)
    order = client.crawled_order
    assert order[0] == 0                       # depth 0 first
    assert set(order[1:]) <= set(range(1, 5)) and len(order) == 4   # 3 expansions (expand_per_node) at depth 1, no depth 2
    assert store.crawl_state("did:plc:0")["followers_at"]
    assert store.one("SELECT COUNT(*) FROM follows")[0] > 0
    assert store.one("SELECT n FROM interactions WHERE src='did:plc:2' AND dst='did:plc:3' AND kind='reply'")[0] == 1
    # persisted: a fresh Store on the same file sees the same graph and does not re-crawl fresh nodes
    store2 = Store(store.path)
    assert store2.counts()["follows"] == store.counts()["follows"]
    crawler2 = Crawler(store2, client, Settings(max_depth=1, ttl_hours=1))
    n_calls = client.calls
    crawler2.request(["u0.test"])
    assert store2.queue_summary()["total"] == 0 and client.calls == n_calls   # cached, no API call


def test_bfs_priority_over_depth(env):
    store, client, crawler = env
    store.enqueue("did:plc:1", priority=2, depth=1)
    store.enqueue("did:plc:0", priority=0, depth=0)
    store.enqueue("did:plc:6", priority=1, depth=0)
    assert [store.dequeue()["did"] for _ in range(3)] == ["did:plc:0", "did:plc:6", "did:plc:1"]


def test_scoring_seeds_propagate(env):
    store, client, crawler = env
    crawler.settings.expand_per_node = 10
    crawler.request(["u0.test", "u5.test"])
    drain(crawler)
    scorer = Scorer(store, ScoringSettings(random_seed_count=2))
    store.set_seed("did:plc:0", 0, "manual")     # trusted
    info = scorer.run()
    assert info["converged"] and info["n_nodes"] == 10
    # rule seeds: 9 spam label -> untrusted; 0 verified -> trusted (manual set above still wins: source stays manual)
    assert store.one("SELECT label, source FROM seeds WHERE did='did:plc:9'")["source"] == "labeler"
    assert store.one("SELECT source FROM seeds WHERE did='did:plc:0'")["source"] == "manual"
    assert store.one("SELECT label, source FROM seeds WHERE did='did:plc:6'")["source"] == "rule:new_account"
    assert store.one("SELECT label, source FROM seeds WHERE did='did:plc:7'")["source"] == "rule:few_followers"
    assert info["rule_seeds"]["labeler"] == 1
    t = {d: store.score(d)["trust"] for d in scorer.graph.dids}
    assert t["did:plc:0"] > 0.5 and t["did:plc:9"] < 0.5
    assert t["did:plc:1"] > t["did:plc:8"]          # trusted clique above the spam clique
    ex = scorer.explain("did:plc:1")
    assert abs(0.5 + sum(r[1] for r in ex["why"]) - ex["trust"]) < 1e-6   # rows sum exactly to the score
    chosen = scorer.draw_random_seeds(2, rng_seed=1)
    assert len(chosen) == 2 and all(store.one("SELECT source FROM seeds WHERE did=?", d)["source"] == "random" for d in chosen)


def test_api_roundtrip(env):
    from fastapi.testclient import TestClient
    from trustscore.bluesky.api import Service, create_app
    store, client, crawler = env
    svc = Service(db_path=store.path, client=client, autostart=False)
    app = create_app(svc)
    tc = TestClient(app)
    r = tc.post("/bsky/lookup", json={"idents": ["u0.test", "@u5.test"], "priority": 0}).json()
    assert r["accounts"]["u0.test"]["state"] == "queued" and r["accounts"]["u5.test"]["did"] == "did:plc:5"
    drain(svc.crawler)
    assert tc.post("/bsky/seeds", json={"action": "set", "ident": "u0.test", "label": 0}).status_code == 200
    r = tc.post("/bsky/lookup", json={"idents": ["u0.test"]}).json()
    assert r["accounts"]["u0.test"]["state"] == "crawled" and r["accounts"]["u0.test"]["score"] > 0.5
    ex = tc.get("/bsky/explain/u1.test").json()
    assert ex["why"] and ex["neighbours"] and ex["facts"]
    assert tc.post("/bsky/seeds", json={"action": "random", "k": 2}).json()["chosen"]
    s = tc.post("/bsky/settings", json={"crawler": {"max_depth": 2}, "scoring": {"alpha": 0.7}}).json()
    assert s["crawler"]["max_depth"] == 2 and s["scoring"]["alpha"] == 0.7
    assert tc.get("/bsky/ego/u0.test").json()["edges"]


def test_local_prior_and_sparse(env):
    from trustscore.bluesky.local_prior import profile_evidence
    p_new, comps = profile_evidence({"handle": "user12345.bsky.social", "display_name": "", "created_at": "2026-09-10T00:00:00Z",
                                     "avatar": None, "followers_count": 3, "follows_count": 900, "posts_count": 0, "labels": "[]"})
    p_old, _ = profile_evidence({"handle": "example.com", "display_name": "Ex", "created_at": "2023-01-01T00:00:00Z",
                                 "avatar": "x", "followers_count": 5000, "follows_count": 100, "posts_count": 300, "labels": None})
    assert 0.9 < p_new <= 1 and 0 <= p_old < 0.4 and len(comps) >= 4
    assert profile_evidence({"did": "did:plc:x", "handle": None})[0] == 0.5     # nothing known -> neutral
    # sparse flag through the API: a leaf seen once is reported as "not enough graph"
    from fastapi.testclient import TestClient
    from trustscore.bluesky.api import Service, create_app
    store, client, crawler = env
    svc = Service(db_path=store.path, client=client, autostart=False)
    tc = TestClient(create_app(svc))
    tc.post("/bsky/lookup", json={"idents": ["u0.test"]})
    drain(svc.crawler)
    svc.rescore()
    r = tc.post("/bsky/lookup", json={"idents": ["u0.test", "u8.test"]}).json()["accounts"]
    assert r["u0.test"]["sparse"] is False and r["u0.test"]["score"] is not None
    assert r["u8.test"]["sparse"] is True and r["u8.test"]["n_edges"] is not None
    svc.rescore()
    ex = tc.get("/bsky/explain/u8.test").json()
    assert any("Not enough graph" in f[0] for f in ex["flags"]) and "profile prior" in ex["why"][0][2]


def test_seed_rules_and_list_import(env):
    from trustscore.bluesky.trusted_lists import import_trusted
    store, client, crawler = env
    crawler.request(["u0.test", "u5.test"])
    drain(crawler)
    scorer = Scorer(store, ScoringSettings())
    scorer.run()
    src = {r["did"]: (r["label"], r["source"]) for r in store.q("SELECT did, label, source FROM seeds")}
    assert src["did:plc:0"] == (0, "rule:verified") and src["did:plc:6"] == (1, "rule:new_account")
    assert src["did:plc:7"] == (1, "rule:few_followers") and src["did:plc:9"] == (1, "labeler")
    # rules can be switched off and are re-derived each run
    scorer.settings.rule_bots = False
    scorer.run()
    assert "did:plc:6" not in {r["did"] for r in store.q("SELECT did FROM seeds")}
    # curated list import: resolves handles, marks them trusted (source list), queues a crawl, keeps manual seeds
    store.set_seed("did:plc:3", 1, "manual")
    rep = import_trusted(store, client, crawler, expand_search=False, backfill=False, handles=["u2.test", "u3.test"])
    assert rep["outlets_resolved"] == 2
    assert store.one("SELECT label, source FROM seeds WHERE did='did:plc:2'")["source"] == "list"
    assert store.one("SELECT label, source FROM seeds WHERE did='did:plc:3'")["source"] == "manual"
    assert store.queued("did:plc:2") or store.is_fresh("did:plc:2", 3600)
