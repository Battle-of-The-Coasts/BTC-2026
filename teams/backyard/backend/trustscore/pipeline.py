"""End-to-end pipeline: dataset -> features -> local model -> propagation -> clusters -> evaluation -> artifacts.

Artifacts written to data/processed/<dataset>/ and served by the API:
  summary.json, metrics.json, clusters.json, graph.json, nodes.parquet, features.parquet,
  local_explanations.json, propagation.npz (channels + per-fold prior/posterior vectors for what-if & explanations)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import roc_curve

from .clusters import cluster_graph, community_report, creation_bursts, dense_blocks, leiden_communities
from .config import Config
from .datasets import load_dataset
from .evaluate import Evaluator, metrics, summarize
from .features import assemble_features
from .graph import build_channels
from .layout import build_display_graph

log = logging.getLogger(__name__)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serialisable: {type(o)}")


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, default=_json_default)


def _roc_points(y, s, n=120):
    ok = np.isfinite(s)
    fpr, tpr, _ = roc_curve(y[ok], s[ok])
    if len(fpr) > n:
        idx = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return [[float(a), float(b)] for a, b in zip(fpr, tpr)]


def _hist(values, bins=20):
    h, edges = np.histogram(values[np.isfinite(values)], bins=bins, range=(0, 1))
    return {"counts": h.tolist(), "edges": edges.tolist()}


def run(cfg: Config, raw_dir: str, out_root: str, skip_sweeps: bool = False) -> str:
    t0 = time.time()
    out = os.path.join(out_root, cfg.dataset)
    os.makedirs(out, exist_ok=True)
    timings = {}

    gd = load_dataset(cfg.dataset, raw_dir, cfg.data)
    timings["load"] = time.time() - t0

    t = time.time()
    X_local, X_all, blocks = assemble_features(gd, burst_window_days=cfg.clusters.burst_window_days,
                                               use_content=cfg.local_model.use_content_features)
    timings["features"] = time.time() - t

    t = time.time()
    channels = build_channels(gd, hub_damping=cfg.propagation.hub_damping)
    timings["channels"] = time.time() - t

    ev = Evaluator(gd, channels, X_local, X_all, cfg)
    lab, y_lab = ev.lab, gd.y[ev.lab]

    # ---- cross-fitted scores for display (every labelled node scored out-of-fold) --------------------------
    t = time.time()
    fold_store = []

    oof, aux = _crossfit_with_store(ev, fold_store, cfg)
    timings["crossfit"] = time.time() - t
    trust = oof["trust_raw"]            # formula score 1/2 - r (display, explanations)
    p_bot = oof["trust_score"]          # calibrated P(bot) (decisions, F1)
    trust_prop = oof["trust_prop_raw"]
    p_local = oof["local_rf"]
    fold_of = aux["fold_of"]

    oof["trust_score_uncal"] = 1 - oof["trust_raw"]   # same weights / alpha / lam, raw rule trust < 1/2 (r > 0), no Platt
    oof_metrics = {m: metrics(y_lab, oof[m][lab]) for m in ("local_rf", "propagation_only", "trust_score", "trust_score_uncal")}
    per_subset = {}
    for s in np.unique(gd.subset[lab]):
        msk = gd.subset[lab] == s
        per_subset[str(s)] = {"n": int(msk.sum()), "label": int(y_lab[msk][0]) if len(np.unique(y_lab[msk])) == 1 else None,
                              "mean_trust": float(trust[lab][msk].mean()),
                              "flagged_share": float((p_bot[lab][msk] > 0.5).mean())}

    # ---- clusters --------------------------------------------------------------------------------------------
    t = time.time()
    W, lab_c, cluster_mode = cluster_graph(gd, cfg.clusters, cfg.propagation.relation_weights)
    membership = leiden_communities(W, resolution=cfg.clusters.leiden_resolution, seed=cfg.clusters.leiden_seed)
    burst_flag, burst_days = creation_bursts(gd.profile, min_accounts=cfg.clusters.burst_min_accounts)
    communities = community_report(gd, W, lab_c, membership, trust, p_bot, burst_flag if len(burst_flag) else None)
    blocks_raw = dense_blocks(gd, n_blocks=cfg.clusters.fraudar_blocks)
    dense = []
    for b in blocks_raw:
        mem = b["members"]
        dense.append({"id": b["id"], "size": int(len(mem)), "n_targets": int(len(b["targets"])), "density": float(b["density"]),
                       "mean_trust": float(trust[mem].mean()), "predicted_bot_share": float((p_bot[mem] > 0.5).mean()),
                       "true_bot_share": float((gd.y[mem] == 1).mean()),
                       "subsets": {str(k): int(v) for k, v in zip(*np.unique(gd.subset[mem], return_counts=True))},
                       "members": [int(i) for i in mem], "top_targets": _top_targets(gd, mem, b["targets"])})
    timings["clusters"] = time.time() - t

    # ---- evaluation sweeps -----------------------------------------------------------------------------------
    sweep_rows, noise_rows = [], []
    if not skip_sweeps:
        t = time.time()
        sweep_rows = ev.seed_sweep()
        noise_rows = ev.noise_sweep()
        timings["sweeps"] = time.time() - t

    # ---- feature importances (fold-0 model, permutation on its held-out fold) --------------------------------
    f0 = fold_store[0]
    test0 = lab[fold_of == 0]
    rows0 = ev.local_pos.loc[test0].to_numpy()
    importances = f0["local_model"].importances(X_local.iloc[rows0], gd.y[test0], n_repeats=5)

    # ---- display graph & layout ------------------------------------------------------------------------------
    t = time.time()
    disp = build_display_graph(gd, W, lab_c, hub_count=cfg.clusters.hub_display_count,
                               sim_top_k=cfg.clusters.display_similarity_top_k)
    timings["layout"] = time.time() - t

    # ---- node table ------------------------------------------------------------------------------------------
    A = gd.relations["follow"].tocsr() if "follow" in gd.relations else None
    if A is not None:
        Ab = A > 0
        out_deg = np.asarray(Ab.sum(axis=1)).ravel()
        in_deg = np.asarray(Ab.sum(axis=0)).ravel()
        mutual_deg = np.asarray(Ab.minimum(Ab.T.tocsr()).sum(axis=1)).ravel()
    else:
        out_deg = in_deg = mutual_deg = np.zeros(gd.n, int)
    comm = np.full(gd.n, -1, int)
    comm[lab_c] = membership
    block_of = np.full(gd.n, -1, int)
    for b in dense:
        block_of[b["members"]] = b["id"]
    burst_all = np.zeros(gd.n, bool)
    if len(burst_flag):
        burst_all[gd.profile.index.to_numpy()] = burst_flag
    display_mask = np.zeros(gd.n, bool)
    display_mask[disp["nodes"]] = True
    xy = np.full((gd.n, 2), np.nan, np.float32)
    xy[disp["nodes"], 0] = disp["x"]
    xy[disp["nodes"], 1] = disp["y"]
    nodes = pd.DataFrame({
        "idx": np.arange(gd.n), "id": gd.node_ids.astype(str), "label": gd.y.astype(int), "subset": gd.subset.astype(str),
        "trust": trust.astype(np.float32), "p_bot": p_bot.astype(np.float32), "trust_prop_only": trust_prop.astype(np.float32),
        "p_local": p_local.astype(np.float32), "in_deg": in_deg, "out_deg": out_deg, "mutual_deg": mutual_deg,
        "community": comm, "block": block_of, "burst": burst_all, "display": display_mask, "x": xy[:, 0], "y": xy[:, 1],
    })
    fold_all = np.full(gd.n, -1, int)
    fold_all[lab] = fold_of
    nodes["fold"] = fold_all
    if gd.profile is not None:
        prof = gd.profile
        for col in ("screen_name", "name", "created_at", "followers_count", "friends_count", "statuses_count"):
            if col in prof:
                s = pd.Series(prof[col].to_numpy(), index=prof.index)
                nodes[col] = s.reindex(nodes.index).astype(object).where(lambda v: v.notna(), None)
    nodes.to_parquet(os.path.join(out, "nodes.parquet"), index=False)
    X_local.to_parquet(os.path.join(out, "features.parquet"))

    # ---- local explanations (Saabas contributions from the model that scored each node out-of-fold) ----------
    expl = {}
    for f in fold_store:
        expl.update(f["explanations"])
    _dump(expl, os.path.join(out, "local_explanations.json"))

    # ---- propagation state for what-if / explanations --------------------------------------------------------
    npz = {}
    for name, G in channels.items():
        Gc = G.tocsr()
        npz[f"ch_{name}_indptr"], npz[f"ch_{name}_indices"], npz[f"ch_{name}_data"] = Gc.indptr, Gc.indices, Gc.data.astype(np.float32)
    for f in fold_store:
        k = f["fold"]
        npz[f"fold{k}_q"] = f["q"].astype(np.float32)
        npz[f"fold{k}_r"] = f["r"].astype(np.float32)
        npz[f"fold{k}_p_local"] = f["p_local"].astype(np.float32)
        npz[f"fold{k}_params"] = np.array([f["alpha"], f["lam"], f["platt_a"], f["platt_b"]], dtype=np.float64)
    npz["fold_of"] = fold_of
    npz["lab"] = lab
    npz["y"] = gd.y
    npz["n"] = np.array([gd.n])
    np.savez_compressed(os.path.join(out, "propagation.npz"), **npz)

    # ---- graph.json for the UI --------------------------------------------------------------------------------
    gnodes = []
    for i, v in enumerate(disp["nodes"]):
        rec = {"i": int(v), "x": float(disp["x"][i]), "y": float(disp["y"][i]), "t": round(float(trust[v]), 4), "p": round(float(p_bot[v]), 4),
               "l": int(gd.y[v]), "s": str(gd.subset[v]), "c": int(comm[v]), "b": int(block_of[v]),
               "d": int(in_deg[v] + out_deg[v]), "id": str(gd.node_ids[v])}
        if gd.profile is not None and v in gd.profile.index:
            rec["n"] = str(gd.profile.loc[v, "screen_name"])
        gnodes.append(rec)
    gedges = [[int(s), int(t), ty, round(w, 3)] for s, t, ty, w in disp["edges"]]
    _dump({"nodes": gnodes, "edges": gedges, "edge_types": sorted({e[2] for e in disp["edges"]})}, os.path.join(out, "graph.json"))

    # ---- clusters.json ---------------------------------------------------------------------------------------
    _dump({"mode": cluster_mode, "resolution": cfg.clusters.leiden_resolution, "n_communities": int(len(communities)),
           "decision_rule": "predicted bot <=> calibrated P(bot) > 0.5",
           "communities": communities, "dense_blocks": dense, "burst_days": burst_days,
           "burst_min_accounts": cfg.clusters.burst_min_accounts}, os.path.join(out, "clusters.json"))

    # ---- metrics.json ----------------------------------------------------------------------------------------
    y_all = y_lab
    metrics_out = {
        "crossfit": {m: {**oof_metrics[m], "roc": _roc_points(y_all, oof[m][lab])} for m in oof_metrics},
        "per_subset": per_subset,
        "seed_sweep": summarize(sweep_rows, ("method", "seed_frac")) if sweep_rows else [],
        "noise_sweep": summarize(noise_rows, ("method", "noise")) if noise_rows else [],
        "seed_sweep_raw": sweep_rows, "noise_sweep_raw": noise_rows,
        "histograms": {"bots": _hist(trust[lab][y_lab == 1]), "humans": _hist(trust[lab][y_lab == 0]),
                       "external": _hist(trust[gd.y < 0]) if (gd.y < 0).any() else None},
        "feature_importance": importances, "feature_blocks": blocks,
        "channel_weights": {f["fold"]: f["weights"] for f in fold_store},
        "homophily": fold_store[0]["homophily"],
        "selected_params": {f["fold"]: {"alpha": f["alpha"], "lam": f["lam"], "platt_a": f["platt_a"], "platt_b": f["platt_b"],
                                        "calibrated_on": f["calibrated_on"], "grid": f["grid_results"],
                                        "propagation_only": {k: v for k, v in f["prop_only"].items() if k in ("alpha", "grid_results")}} for f in fold_store},
        "propagation_info": {"iterations": fold_store[0]["iterations"], "converged": fold_store[0]["converged"]},
    }
    _dump(metrics_out, os.path.join(out, "metrics.json"))

    timings["total"] = time.time() - t0
    ds_summary = gd.summary()
    ds_summary["shared_followee_purity"] = _shared_followee_purity(gd)
    summary = {"dataset": ds_summary, "config": cfg.to_dict(), "timings": {k: round(v, 1) for k, v in timings.items()},
               "n_display_nodes": int(len(disp["nodes"])), "n_display_edges": int(len(disp["edges"])),
               "n_features": int(X_local.shape[1]), "feature_blocks": {k: len(v) for k, v in blocks.items()},
               "crossfit_folds": cfg.eval.crossfit_folds}
    _dump(summary, os.path.join(out, "summary.json"))
    log.info("pipeline done in %.1fs -> %s", timings["total"], out)
    return out


def _top_targets(gd, members, targets, k: int = 10) -> list:
    """Targets of a dense block ordered by how many block members follow them."""
    if "follow" not in gd.relations or len(targets) == 0:
        return []
    A = gd.relations["follow"].tocsr()
    cnt = np.asarray((A[members][:, targets] > 0).sum(axis=0)).ravel()
    order = np.argsort(-cnt)[:k]
    return [{"node": int(targets[i]), "followed_by": int(cnt[i]), "share": float(cnt[i] / max(1, len(members)))} for i in order]


def _shared_followee_purity(gd) -> dict | None:
    """Reporting only: among external accounts followed by >= 2 labelled accounts, how label-pure are their followers?"""
    if "follow" not in gd.relations or not (gd.y < 0).any():
        return None
    A = gd.relations["follow"].tocsr()
    lab = gd.labeled_idx
    ext = np.flatnonzero(gd.y < 0)
    B = (A[lab][:, ext] > 0).astype(np.float64)
    n_foll = np.asarray(B.sum(axis=0)).ravel()
    n_bot = np.asarray(B.T @ (gd.y[lab] == 1).astype(np.float64)).ravel()
    m = n_foll >= 2
    if not m.any():
        return None
    share = n_bot[m] / n_foll[m]
    purity = np.maximum(share, 1 - share)
    return {"n_shared_followees": int(m.sum()), "mean_purity": float(purity.mean()), "frac_pure": float(((share == 0) | (share == 1)).mean())}


def _crossfit_with_store(ev: Evaluator, store: list, cfg: Config):
    """Evaluator-style cross-fitting plus per-fold state needed by the API (prior/posterior vectors, local explanations)."""
    gd = ev.gd
    lab = ev.lab
    y_lab = gd.y[lab]
    folds = cfg.eval.crossfit_folds
    rng = np.random.default_rng(cfg.eval.random_state + 77)
    fold_of = np.empty(len(lab), int)
    for cls in (0, 1):
        idx = np.flatnonzero(y_lab == cls)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % folds
    methods = ("local_rf", "propagation_only", "trust_score")
    keys = ("local_rf", "propagation_only", "trust_score", "trust_raw", "trust_prop_raw")
    oof = {m: np.full(gd.n, np.nan) for m in keys}
    fold0 = {}
    for f in range(folds):
        train = np.flatnonzero(fold_of != f)
        test = np.flatnonzero(fold_of == f)
        scores, extras = ev.run_all(lab[train], y_lab[train], list(methods), random_state=f)
        ts, po = extras["trust_score"], extras["propagation_only"]
        vec = {"local_rf": scores["local_rf"], "propagation_only": scores["propagation_only"], "trust_score": scores["trust_score"],
               "trust_raw": ts.T, "trust_prop_raw": po.T}
        for m in keys:
            oof[m][lab[test]] = vec[m][lab[test]]
            if f == 0:
                fold0[m] = vec[m]
        model = extras["local_model"]
        test_nodes = lab[test]
        rows = ev.local_pos.loc[test_nodes].to_numpy()
        Xt = ev.X_local.iloc[rows]
        bias, C = model.contributions(Xt)
        expl = {}
        names = list(Xt.columns)
        for k, v in enumerate(test_nodes):
            order = np.argsort(-np.abs(C[k]))[:8]
            expl[int(v)] = {"bias": float(bias), "top": [{"feature": names[j], "contribution": float(C[k, j]), "value": float(Xt.iat[k, j])} for j in order]}
        store.append({"fold": f, "q": ts.q, "r": ts.r, "p_local": scores["local_rf"], "weights": ts.weights,
                      "homophily": ts.homophily, "iterations": ts.iterations, "converged": ts.converged,
                      "alpha": ts.alpha, "lam": ts.lam, "platt_a": ts.platt_a, "platt_b": ts.platt_b,
                      "grid_results": ts.grid_results, "calibrated_on": ts.calibrated_on,
                      "prop_only": po.export(), "local_model": model, "explanations": expl})
        log.info("crossfit fold %d/%d done (alpha=%.2f lam=%.2f)", f + 1, folds, ts.alpha, ts.lam)
    ext = gd.y < 0   # unlabelled accounts: scored by fold 0 (80 % of the labels), the same run the API explains
    for m in keys:
        oof[m][ext] = fold0[m][ext]
    return oof, {"fold_of": fold_of}
