"""FastAPI server exposing the precomputed artifacts plus two live computations:
* per-node explanation (prior term, per-channel and per-neighbour propagation terms, local feature contributions)
* what-if re-propagation with different alpha / lambda / channel weights on the fold-0 split.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import PropagationConfig
from .evaluate import metrics
from .graph import combine_channels
from .propagation import build_prior, explain_node, propagate, trust_from_residual

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED = os.environ.get("TRUSTSCORE_PROCESSED", os.path.join(ROOT, "data", "processed"))
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")

app = FastAPI(title="Trust Score API", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=2000)


class DatasetStore:
    def __init__(self, name: str):
        self.name = name
        self.dir = os.path.join(PROCESSED, name)
        if not os.path.isdir(self.dir):
            raise HTTPException(404, f"dataset {name} not processed")
        self.summary = self._json("summary.json")
        self.metrics = self._json("metrics.json")
        self.clusters = self._json("clusters.json")
        self.graph = self._json("graph.json")
        self.local_expl = self._json("local_explanations.json")
        self.nodes = pd.read_parquet(os.path.join(self.dir, "nodes.parquet"))
        self.features = pd.read_parquet(os.path.join(self.dir, "features.parquet"))
        self.feature_pct = self.features.rank(pct=True)
        z = np.load(os.path.join(self.dir, "propagation.npz"))
        n = int(z["n"][0])
        self.n = n
        self.channels: dict[str, sp.csr_matrix] = {}
        for k in z.files:
            if k.startswith("ch_") and k.endswith("_indptr"):
                name_ = k[3:-7]
                self.channels[name_] = sp.csr_matrix((z[f"ch_{name_}_data"], z[f"ch_{name_}_indices"], z[k]), shape=(n, n))
        self.folds = {}
        f = 0
        while f"fold{f}_q" in z.files:
            params = z[f"fold{f}_params"] if f"fold{f}_params" in z.files else None
            self.folds[f] = {"q": z[f"fold{f}_q"].astype(np.float64), "r": z[f"fold{f}_r"].astype(np.float64),
                             "p_local": z[f"fold{f}_p_local"].astype(np.float64),
                             "alpha": float(params[0]) if params is not None else None, "lam": float(params[1]) if params is not None else None,
                             "platt_a": float(params[2]) if params is not None else 1.0, "platt_b": float(params[3]) if params is not None else 0.0}
            f += 1
        self.fold_of = z["fold_of"]
        self.lab = z["lab"]
        self.y = z["y"]
        self.cfg = PropagationConfig(**{k: v for k, v in self.summary["config"]["propagation"].items()})
        self.weights_by_fold = {int(k): v for k, v in self.metrics["channel_weights"].items()}
        self.node_fold = dict(zip(self.lab.tolist(), self.fold_of.tolist()))
        self.community_of = {}
        for c in self.clusters["communities"]:
            for m in c["members"]:
                self.community_of[m] = c["id"]

    def _json(self, name):
        with open(os.path.join(self.dir, name)) as f:
            return json.load(f)

    def fold_for(self, v: int) -> int:
        return int(self.node_fold.get(int(v), 0))


@lru_cache(maxsize=4)
def store(name: str) -> DatasetStore:
    return DatasetStore(name)


def _clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = None if np.isnan(v) else float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


@app.get("/api/datasets")
def datasets():
    out = []
    if os.path.isdir(PROCESSED):
        for d in sorted(os.listdir(PROCESSED)):
            p = os.path.join(PROCESSED, d, "summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    out.append({"name": d, "summary": json.load(f)})
    return out


@app.get("/api/docs")
def docs():
    from fastapi.responses import PlainTextResponse
    path = os.path.join(ROOT, "docs", "DESIGN.md")
    if not os.path.exists(path):
        raise HTTPException(404, "docs/DESIGN.md not found")
    with open(path) as f:
        return PlainTextResponse(f.read(), media_type="text/markdown")


@app.get("/api/{ds}/summary")
def summary(ds: str):
    return store(ds).summary


@app.get("/api/{ds}/metrics")
def get_metrics(ds: str):
    m = dict(store(ds).metrics)
    m.pop("seed_sweep_raw", None)
    m.pop("noise_sweep_raw", None)
    return m


@app.get("/api/{ds}/graph")
def graph(ds: str):
    return store(ds).graph


@app.get("/api/{ds}/clusters")
def clusters(ds: str):
    c = store(ds).clusters
    comms = [{k: v for k, v in x.items() if k != "members"} | {"members_preview": x["members"][:12]} for x in c["communities"]]
    blocks = [{k: v for k, v in x.items() if k != "members"} | {"members_preview": x["members"][:12]} for x in c["dense_blocks"]]
    return {**c, "communities": comms, "dense_blocks": blocks}


@app.get("/api/{ds}/cluster/{cid}/members")
def cluster_members(ds: str, cid: int, kind: str = "community"):
    c = store(ds).clusters
    items = c["communities"] if kind == "community" else c["dense_blocks"]
    for x in items:
        if x["id"] == cid:
            return {"id": cid, "members": x["members"]}
    raise HTTPException(404, "cluster not found")


@app.get("/api/{ds}/nodes")
def nodes(ds: str, q: str | None = None, label: int | None = None, subset: str | None = None,
          community: int | None = None, block: int | None = None, only_labeled: bool = True,
          min_trust: float | None = None, max_trust: float | None = None,
          sort: str = "trust", order: str = "asc", limit: int = Query(50, le=500), offset: int = 0):
    s = store(ds)
    df = s.nodes
    if only_labeled:
        df = df[df.label >= 0]
    if q:
        ql = q.lower()
        m = df.id.str.lower().str.contains(ql, regex=False)
        if "screen_name" in df:
            m |= df.screen_name.fillna("").astype(str).str.lower().str.contains(ql, regex=False)
        df = df[m]
    if label is not None:
        df = df[df.label == label]
    if subset:
        df = df[df.subset == subset]
    if community is not None:
        df = df[df.community == community]
    if block is not None:
        df = df[df.block == block]
    if min_trust is not None:
        df = df[df.trust >= min_trust]
    if max_trust is not None:
        df = df[df.trust <= max_trust]
    if sort in df.columns:
        df = df.sort_values(sort, ascending=(order == "asc"), kind="stable")
    total = int(len(df))
    page = df.iloc[offset: offset + limit]
    cols = [c for c in page.columns if c not in ("x", "y", "display")]
    return {"total": total, "rows": [_clean(r) for r in page[cols].to_dict("records")]}


@app.get("/api/{ds}/node/{idx}")
def node(ds: str, idx: int, top_k: int = 12):
    s = store(ds)
    if idx < 0 or idx >= s.n:
        raise HTTPException(404, "node not found")
    row = _clean(s.nodes.iloc[idx].to_dict())
    f = s.fold_for(idx)
    fold = s.folds[f]
    weights = s.weights_by_fold.get(f, s.cfg.relation_weights)
    alpha = fold["alpha"] if fold.get("alpha") is not None else s.cfg.alpha
    lam = fold["lam"] if fold.get("lam") is not None else s.cfg.lam
    ex = explain_node(idx, s.channels, weights, fold["q"], fold["r"], alpha, top_k=top_k)
    ex["p_bot"] = float(1 / (1 + np.exp(-np.clip(fold["platt_a"] * fold["r"][idx] + fold["platt_b"], -60, 60))))
    for nb in ex["top_neighbors"]:
        u = nb["node"]
        r = s.nodes.iloc[u]
        # trust of the neighbour *in the explaining fold* (the value that actually entered this node's score);
        # the table's out-of-fold trust of the neighbour is given separately
        nb["label"] = int(r["label"]); nb["trust"] = float(np.clip(0.5 - fold["r"][u], 0, 1)); nb["trust_oof"] = float(r["trust"]); nb["id"] = str(r["id"])
        nb["screen_name"] = None if "screen_name" not in r or pd.isna(r["screen_name"]) else str(r["screen_name"])
        nb["is_seed"] = bool(r["label"] >= 0 and s.node_fold.get(u, -1) != f)
    feats = None
    if idx in s.features.index:
        vals = s.features.loc[idx]
        pct = s.feature_pct.loc[idx]
        feats = {c: {"value": float(vals[c]), "pct": float(pct[c])} for c in s.features.columns}
    local = s.local_expl.get(str(idx))
    return {"node": row, "fold": f, "propagation": ex, "features": feats, "local_explanation": local,
            "community": s.community_of.get(idx), "alpha": alpha, "lam": lam, "weights": weights,
            "platt": {"a": fold["platt_a"], "b": fold["platt_b"]}}


class WhatIf(BaseModel):
    alpha: float | None = None
    lam: float | None = None
    weights: dict[str, float] | None = None
    fold: int = 0


@app.post("/api/{ds}/whatif")
def whatif(ds: str, body: WhatIf):
    s = store(ds)
    f = body.fold if body.fold in s.folds else 0
    fd = s.folds[f]
    alpha = (fd["alpha"] if fd.get("alpha") is not None else s.cfg.alpha) if body.alpha is None else float(np.clip(body.alpha, 0.0, 0.99))
    lam = (fd["lam"] if fd.get("lam") is not None else s.cfg.lam) if body.lam is None else float(np.clip(body.lam, 0.0, 1.0))
    weights = dict(s.weights_by_fold.get(f, s.cfg.relation_weights))
    if body.weights:
        weights.update({k: float(np.clip(v, 0.0, 1.0)) for k, v in body.weights.items() if k in weights})
    train = s.lab[s.fold_of != f]
    test = s.lab[s.fold_of == f]
    q = build_prior(s.n, train, s.y[train], s.folds[f]["p_local"], lam=lam, seed_residual=s.cfg.seed_residual)
    P = combine_channels(s.channels, weights)
    r, info = propagate(P, q, alpha=alpha, max_iter=s.cfg.max_iter, tol=s.cfg.tol, return_info=True)
    T = trust_from_residual(r)
    p_bot = 1 / (1 + np.exp(-np.clip(fd["platt_a"] * r + fd["platt_b"], -60, 60)))   # fold calibration (fitted for the pipeline's parameters)
    m = metrics(s.y[test], p_bot[test])
    disp = s.nodes.index[s.nodes.display].to_numpy()
    return {"alpha": alpha, "lam": lam, "weights": weights, "fold": f, "iterations": info["iterations"],
            "metrics": m, "trust": {int(i): round(float(T[i]), 4) for i in disp}, "p_bot": {int(i): round(float(p_bot[i]), 4) for i in disp}}


if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
