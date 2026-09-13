"""FastAPI app for the Bluesky add-on. Runs on its own port (default 8010) next to the dataset dashboard.

Endpoints (all JSON, CORS open so the content script can call from https://bsky.app):
  GET  /bsky/health
  GET  /bsky/status                       crawler + store + last run
  POST /bsky/lookup  {idents:[..], priority:0|1}   scores for known accounts, enqueue the rest (BFS)
  GET  /bsky/explain/{ident}              popover payload (facts, why-rows, neighbours, flags)
  GET  /bsky/ego/{ident}?limit=60         small ego graph
  GET  /bsky/analytics                    histograms, bands, degrees, channels, runs, top lists (dashboard Overview)
  GET  /bsky/graph?focus=a,b&hops=1&max_nodes=1500   node/edge lists for the dashboard Network view
  GET  /bsky/accounts?q=&sort=trust&desc=1&limit=100&offset=0   crawled accounts table
  GET  /bsky/seeds  |  POST /bsky/seeds   {action: random|set|remove|clear, ident, label, k}
  GET  /bsky/settings | POST /bsky/settings   crawler + scoring parameters
  POST /bsky/recompute                    run the propagation now
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analytics import Analytics
from .client import BskyClient
from .crawler import Crawler, Settings
from .scoring import Scorer, ScoringSettings
from .store import Store
from .trusted_lists import import_trusted

log = logging.getLogger("bluesky.api")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DB_PATH = os.environ.get("BLUESKY_DB", os.path.join(ROOT, "data", "bluesky", "bluesky.sqlite"))


class Service:
    def __init__(self, db_path: str = DB_PATH, client: BskyClient | None = None, autostart: bool = True):
        self.store = Store(db_path)
        self.crawler_settings = Settings.load(self.store)
        self.client = client or BskyClient(per_second=self.crawler_settings.requests_per_second)
        self.scorer = Scorer(self.store, ScoringSettings.load(self.store))
        self.analytics = Analytics(self.store)
        self.crawler = Crawler(self.store, self.client, self.crawler_settings, on_change=self._changed)
        self._dirty = False
        self._last_score = 0.0
        self._score_lock = threading.Lock()
        if autostart:
            self.crawler.start()
            if self.store.last_run():
                try:
                    self.rescore()
                except Exception:
                    log.exception("initial rescore failed")

    def _changed(self):
        self._dirty = True
        if time.time() - self._last_score > self.crawler_settings.rescore_every_seconds:
            self.rescore()

    def rescore(self) -> dict:
        with self._score_lock:
            self._dirty = False
            self._last_score = time.time()
            return self.scorer.run()

    def maybe_rescore(self):
        if self._dirty and time.time() - self._last_score > self.crawler_settings.rescore_every_seconds:
            self.rescore()

    # ---- payloads ------------------------------------------------------------------------------
    def account_payload(self, did: str) -> dict:
        a = self.store.actor(did)
        sc = self.store.score(did)
        c = self.store.crawl_state(did)
        seed = self.store.one("SELECT label, source FROM seeds WHERE did=?", did)
        labels = json.loads(a["labels"]) if a and a["labels"] else []
        crawled = bool(c and c["followers_at"])
        state = "crawled" if crawled else ("queued" if self.store.queued(did) else ("error" if c and c["error"] else "pending"))
        deg = 0
        if not crawled:
            deg = int(self.store.one("SELECT (SELECT COUNT(*) FROM follows WHERE src=? OR dst=?) + "
                                     "(SELECT COUNT(*) FROM interactions WHERE src=? OR dst=?)", did, did, did, did)[0])
        sparse = (not crawled) and deg < self.scorer.settings.min_neighbours_for_score
        return {
            "sparse": sparse, "n_edges": deg if not crawled else None, "verified": a["verified"] if a else None,
            "did": did, "handle": a["handle"] if a else None, "displayName": a["display_name"] if a else None,
            "score": float(sc["trust"]) if sc else None, "state": state, "depth": a["depth"] if a else None,
            "seed": {"label": seed["label"], "source": seed["source"]} if seed else None, "labels": labels,
            "followers": a["followers_count"] if a else None, "follows": a["follows_count"] if a else None,
            "posts": a["posts_count"] if a else None, "createdAt": a["created_at"] if a else None,
        }

    def explain_payload(self, did: str) -> dict:
        p = self.account_payload(did)
        ex = self.scorer.explain(did)
        nb = self.store.neighbours(did)
        c = self.store.crawl_state(did)
        facts = []
        if p["createdAt"]:
            try:
                dt = datetime.fromisoformat(p["createdAt"].replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - dt).days
                facts.append(["Account age", f"{age} days" if age < 400 else f"{age / 365.25:.1f} years"])
            except ValueError:
                pass
        for k, lab in (("followers", "Followers"), ("follows", "Following"), ("posts", "Posts")):
            if p[k] is not None:
                facts.append([lab, f"{p[k]:,}"])
        mutual = nb["follows"] & nb["followers"]
        facts.append(["In the crawled graph", f"{len(nb['followers'])} followers · {len(nb['follows'])} follows · {len(mutual)} mutual"])
        facts.append(["Interactions seen", f"{len(nb['interacts_with'])} out · {len(nb['interacted_by'])} in"])
        flags = []
        if p["seed"]:
            flags.append([("Untrusted seed" if p["seed"]["label"] == 1 else "Trusted seed") + f" ({p['seed']['source']})",
                          "ts-red" if p["seed"]["label"] == 1 else "ts-green"])
        if p["verified"]:
            flags.append(["Trusted verifier" if p["verified"] == 2 else "Verified account", "ts-green"])
        for l in p["labels"]:
            flags.append([f"Moderation label: {l}", "ts-red"])
        if c and (c["followers_truncated"] or c["follows_truncated"]):
            flags.append(["Lists truncated by the crawl cap", "ts-grey"])
        if p["state"] != "crawled":
            flags.append([f"Crawl {p['state']}", "ts-amber"])
        if p["sparse"]:
            flags.append([f"Not enough graph: {p['n_edges']} stored edge(s), score is the profile prior only", "ts-grey"])
        out = {**p, "facts": facts, "flags": flags, "why": ex["why"] if ex else [],
               "neighbours": ex["neighbours"] if ex else [], "computedAt": ex["computed_at"] if ex else None}
        return out

    def ego(self, did: str, limit: int = 60) -> dict:
        nb = self.store.neighbours(did)
        neigh = list((nb["follows"] | nb["followers"] | nb["interacts_with"] | nb["interacted_by"]))[:limit]
        ids = [did] + neigh
        nodes = [self.account_payload(d) for d in ids]
        s = set(ids)
        edges = []
        for r in self.store.q(f"SELECT src, dst, 'follow' k, 1 n FROM follows WHERE src IN ({','.join('?' * len(ids))}) "
                              f"UNION ALL SELECT src, dst, kind, n FROM interactions WHERE src IN ({','.join('?' * len(ids))})",
                              *ids, *ids):
            if r["dst"] in s:
                edges.append({"src": r["src"], "dst": r["dst"], "kind": r["k"], "n": r["n"]})
        return {"nodes": nodes, "edges": edges}


class LookupBody(BaseModel):
    idents: list[str]
    priority: int = 0


class SeedBody(BaseModel):
    action: str
    ident: str | None = None
    label: int | None = None
    k: int | None = None
    rng_seed: int | None = None
    source: str | None = None
    expand_search: bool = True
    handles: list[str] | None = None


class SettingsBody(BaseModel):
    crawler: dict | None = None
    scoring: dict | None = None


def create_app(service: Service | None = None) -> FastAPI:
    app = FastAPI(title="Trust Score — Bluesky add-on API", version="0.1")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    svc = service or Service()
    app.state.service = svc
    addon_dir = os.path.join(ROOT, "bluesky-addon")
    if os.path.isdir(addon_dir):   # dev convenience: inject /addon/overlay.css + /addon/content.js into any tab
        app.mount("/addon", StaticFiles(directory=addon_dir), name="addon")

    @app.get("/bsky/health")
    def health():
        return {"ok": True}

    @app.get("/bsky/status")
    def status():
        return {"crawler": svc.crawler.status(), "store": svc.store.counts(), "last_run": svc.store.last_run(),
                "scoring": svc.scorer.settings.__dict__}

    @app.post("/bsky/lookup")
    def lookup(body: LookupBody):
        idents = list(dict.fromkeys(i.strip().lstrip("@") for i in body.idents if i and i.strip()))[:200]
        resolved = svc.crawler.request(idents, priority=max(0, min(2, body.priority)))
        svc.maybe_rescore()
        return {"accounts": {ident: (svc.account_payload(did) if did else None) for ident, did in resolved.items()},
                "run": svc.store.last_run()}

    def _did(ident: str) -> str:
        row = svc.store.resolve(ident.lstrip("@"))
        if row is None:
            res = svc.crawler.request([ident.lstrip("@")], priority=0)
            did = res.get(ident.lstrip("@"))
            if not did:
                raise HTTPException(404, f"unknown account {ident}")
            return did
        return row["did"]

    @app.get("/bsky/explain/{ident}")
    def explain(ident: str):
        return svc.explain_payload(_did(ident))

    @app.get("/bsky/ego/{ident}")
    def ego(ident: str, limit: int = 60):
        return svc.ego(_did(ident), limit)

    @app.get("/bsky/analytics")
    def analytics():
        return {"store": svc.store.counts(), "crawler": svc.crawler.status(), "last_run": svc.store.last_run(),
                "scoring": svc.scorer.settings.__dict__, **svc.analytics.analytics(svc.scorer.settings.channel_weights)}

    @app.get("/bsky/graph")
    def graph(focus: str = "", hops: int = 1, max_nodes: int = 1500, min_degree: int = 2):
        dids = []
        for ident in [x.strip().lstrip("@") for x in focus.split(",") if x.strip()]:
            row = svc.store.resolve(ident)
            if row is None:
                did = svc.crawler.request([ident], priority=0).get(ident)
                if did:
                    dids.append(did)
            else:
                dids.append(row["did"])
        return svc.analytics.graph(dids, hops=max(0, min(hops, 3)), max_nodes=max(10, min(max_nodes, 4000)), min_degree=min_degree)

    @app.get("/bsky/accounts")
    def accounts(q: str | None = None, sort: str = "trust", desc: int = 1, limit: int = 100, offset: int = 0, all: int = 0):
        return svc.analytics.accounts(q, sort, bool(desc), max(1, min(limit, 500)), max(0, offset), only_crawled=not all)

    @app.get("/bsky/seeds")
    def seeds():
        return {"seeds": [dict(r) for r in svc.store.seeds()]}

    @app.post("/bsky/seeds")
    def seeds_post(body: SeedBody):
        if body.action == "random":
            chosen = svc.scorer.draw_random_seeds(body.k, body.rng_seed)
            if body.k:
                svc.scorer.settings.random_seed_count = body.k; svc.scorer.settings.save(svc.store)
            svc.rescore()
            return {"chosen": chosen, "seeds": [dict(r) for r in svc.store.seeds()]}
        if body.action == "import_list":
            rep = import_trusted(svc.store, svc.client, svc.crawler, expand_search=body.expand_search, handles=body.handles)
            info = svc.rescore()
            return {**rep, "rule_seeds": info.get("rule_seeds"), "seeds_by": [dict(r) for r in svc.store.q(
                "SELECT source, label, COUNT(*) n FROM seeds GROUP BY source, label ORDER BY source")]}
        if body.action == "clear_source":
            if not body.source:
                raise HTTPException(400, "source required")
            svc.store.clear_seeds(source=body.source)
        elif body.action == "set":
            if body.ident is None or body.label not in (0, 1):
                raise HTTPException(400, "ident and label (0 trusted / 1 untrusted) required")
            svc.store.set_seed(_did(body.ident), body.label, "manual")
        elif body.action == "remove":
            if body.ident is None:
                raise HTTPException(400, "ident required")
            svc.store.remove_seed(_did(body.ident))
        elif body.action == "clear":
            svc.store.clear_seeds()
        else:
            raise HTTPException(400, "unknown action")
        svc.rescore()
        return {"seeds": [dict(r) for r in svc.store.seeds()]}

    @app.get("/bsky/settings")
    def settings():
        return {"crawler": svc.crawler.settings.__dict__, "scoring": svc.scorer.settings.__dict__}

    @app.post("/bsky/settings")
    def settings_post(body: SettingsBody):
        if body.crawler:
            for k, v in body.crawler.items():
                if k in Settings.__dataclass_fields__:
                    setattr(svc.crawler.settings, k, type(getattr(svc.crawler.settings, k))(v))
            svc.crawler.settings.save(svc.store)
            svc.client.limiter.rate = svc.crawler.settings.requests_per_second
        if body.scoring:
            for k, v in body.scoring.items():
                if k == "channel_weights" and isinstance(v, dict):
                    svc.scorer.settings.channel_weights.update({kk: float(vv) for kk, vv in v.items()})
                elif k in ScoringSettings.__dataclass_fields__:
                    setattr(svc.scorer.settings, k, type(getattr(svc.scorer.settings, k))(v))
            svc.scorer.settings.save(svc.store)
            svc.rescore()
        return settings()

    @app.post("/bsky/recompute")
    def recompute():
        return svc.rescore()

    return app


app = create_app() if os.environ.get("BLUESKY_AUTOAPP", "1") == "1" else None
