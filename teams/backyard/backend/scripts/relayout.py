"""Rebuild only graph.json (+ x/y in nodes.parquet) for a processed dataset — no scoring is redone.
Usage: python scripts/relayout.py cresci-2015"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from trustscore.clusters import cluster_graph
from trustscore.config import Config
from trustscore.datasets import load_dataset
from trustscore.layout import build_display_graph
from trustscore.pipeline import _dump

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    ds = sys.argv[1]
    cfg = Config(dataset=ds)
    out = os.path.join(ROOT, "data", "processed", ds)
    gd = load_dataset(ds, os.path.join(ROOT, "data", "raw"), cfg.data)
    nodes = pd.read_parquet(os.path.join(out, "nodes.parquet"))
    W, lab_c, _ = cluster_graph(gd, cfg.clusters, cfg.propagation.relation_weights)
    disp = build_display_graph(gd, W, lab_c, hub_count=cfg.clusters.hub_display_count, sim_top_k=cfg.clusters.display_similarity_top_k)
    trust = nodes["trust"].to_numpy(); p_bot = nodes["p_bot"].to_numpy() if "p_bot" in nodes else 1 - trust
    comm = nodes["community"].to_numpy(); block_of = nodes["block"].to_numpy()
    deg = (nodes["in_deg"] + nodes["out_deg"]).to_numpy()
    gnodes = []
    for i, v in enumerate(disp["nodes"]):
        rec = {"i": int(v), "x": float(disp["x"][i]), "y": float(disp["y"][i]), "t": round(float(trust[v]), 4), "p": round(float(p_bot[v]), 4),
               "l": int(gd.y[v]), "s": str(gd.subset[v]), "c": int(comm[v]), "b": int(block_of[v]), "d": int(deg[v]), "id": str(gd.node_ids[v])}
        if gd.profile is not None and v in gd.profile.index:
            rec["n"] = str(gd.profile.loc[v, "screen_name"])
        gnodes.append(rec)
    gedges = [[int(s), int(t), ty, round(w, 3)] for s, t, ty, w in disp["edges"]]
    _dump({"nodes": gnodes, "edges": gedges, "edge_types": sorted({e[2] for e in disp["edges"]})}, os.path.join(out, "graph.json"))
    xy = np.full((gd.n, 2), np.nan, np.float32)
    xy[disp["nodes"], 0] = disp["x"]; xy[disp["nodes"], 1] = disp["y"]
    display = np.zeros(gd.n, bool); display[disp["nodes"]] = True
    nodes["x"], nodes["y"], nodes["display"] = xy[:, 0], xy[:, 1], display
    nodes.to_parquet(os.path.join(out, "nodes.parquet"), index=False)
    summ_path = os.path.join(out, "summary.json")
    if os.path.exists(summ_path):
        summ = json.load(open(summ_path)); summ["n_display_nodes"] = int(len(disp["nodes"])); summ["n_display_edges"] = int(len(disp["edges"]))
        _dump(summ, summ_path)
    print("relayout done:", len(gnodes), "nodes", len(gedges), "edges")
