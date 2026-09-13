"""Display graph (labelled accounts + the most-followed external hubs) and its 2-D layout for the UI.

The display graph is a *sketch* for humans, never an input to any score. Candidate edges (co-following similarity,
mutual follows, one-way follows, follows of external hubs) are capped per node, then selected greedily by priority
(similarity > mutual > one-way > hub) under a global budget of ~`avg_degree` edges per node so that 10k-node
graphs stay readable. Positions: DrL (a scalable force-directed layout that separates communities) refined by a
short Fruchterman-Reingold pass, then robustly rescaled so that the bulk of the connected part fills the unit disk
(outliers are squeezed onto its rim); isolated accounts sit on an outer ring.
"""
from __future__ import annotations

import logging

import igraph as ig
import numpy as np
import scipy.sparse as sp

from .datasets import GraphData

log = logging.getLogger(__name__)

PRIORITY = {"similar": 0, "mutual": 1, "follow": 2, "follow_hub": 3}


def _topk_per_row(W: sp.csr_matrix, k: int) -> sp.csr_matrix:
    W = W.tocsr()
    rows, cols, vals = [], [], []
    for i in range(W.shape[0]):
        lo, hi = W.indptr[i], W.indptr[i + 1]
        idx, val = W.indices[lo:hi], W.data[lo:hi]
        if len(val) > k:
            keep = np.argpartition(-val, k)[:k]
            idx, val = idx[keep], val[keep]
        rows.append(np.full(len(idx), i)); cols.append(idx); vals.append(val)
    if not rows:
        return sp.csr_matrix(W.shape, dtype=W.dtype)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=W.shape)


def select_edges(edges: list, n: int, pos: dict, is_hub: np.ndarray, rng: np.random.Generator,
                 avg_degree: float = 6.0, node_cap: int = 10, hub_cap: int = 60) -> list:
    """Greedy edge budget. Edges are visited by (type priority, -weight); an edge is kept while the global budget
    (`avg_degree * n`) is not exhausted and both endpoints are under their cap, or when one endpoint has no display
    edge yet (so that no connected account is drawn as isolated)."""
    order = sorted(range(len(edges)), key=lambda k: (PRIORITY.get(edges[k][2], 9), -edges[k][3], rng.random()))
    deg = np.zeros(n, int)
    keep = []
    budget = int(avg_degree * n)
    for k in order:
        s_, t_, _, _ = edges[k]
        a, b = pos[s_], pos[t_]
        cap_a = hub_cap if is_hub[a] else node_cap
        cap_b = hub_cap if is_hub[b] else node_cap
        first = deg[a] == 0 or deg[b] == 0
        if first or (len(keep) < budget and deg[a] < cap_a and deg[b] < cap_b):
            keep.append(k)
            deg[a] += 1; deg[b] += 1
    return [edges[k] for k in keep]


def layout_positions(n: int, e_idx: list, weights: list, rng: np.random.Generator, fr_iter: int = 120) -> tuple[np.ndarray, int]:
    """(x, y) for every node index in [0, n): DrL + FR refinement on the connected part (rescaled so that the 98th
    percentile radius is 1 and outliers are squeezed onto the rim), outer ring (radius 1.25-1.37) for isolated nodes."""
    deg = np.zeros(n, int)
    for a, b in e_idx:
        deg[a] += 1; deg[b] += 1
    connected = np.flatnonzero(deg > 0)
    isolated = np.flatnonzero(deg == 0)
    xy = np.zeros((n, 2), dtype=np.float32)
    if len(connected):
        remap = np.full(n, -1, int)
        remap[connected] = np.arange(len(connected))
        sub = ig.Graph(n=len(connected), edges=[(int(remap[a]), int(remap[b])) for a, b in e_idx], directed=False)
        w = list(weights) if len(weights) == sub.ecount() else None
        c = np.asarray(sub.layout_drl(weights=w).coords, dtype=np.float64)
        c = (c - c.mean(0)) / (c.std(0) + 1e-9) * (0.25 * np.sqrt(len(connected)))   # ~unit spacing for FR
        if fr_iter > 0:
            c = np.asarray(sub.layout_fruchterman_reingold(weights=w, niter=fr_iter, grid="nogrid", seed=c.tolist()).coords,
                           dtype=np.float64)
        c -= np.median(c, 0)
        r = np.hypot(c[:, 0], c[:, 1])
        scale = float(np.percentile(r, 98)) + 1e-9
        c /= scale; r /= scale
        far = r > 1.0
        c[far] *= ((1.0 + 0.15 * np.tanh(r[far] - 1.0)) / r[far])[:, None]
        xy[connected] = c
    if len(isolated):
        ang = rng.uniform(0, 2 * np.pi, len(isolated))
        rad = 1.25 + 0.12 * rng.random(len(isolated))
        xy[isolated, 0] = rad * np.cos(ang)
        xy[isolated, 1] = rad * np.sin(ang)
    return xy, int(len(isolated))


def build_display_graph(gd: GraphData, W_cluster: sp.csr_matrix, lab: np.ndarray, hub_count: int = 150,
                        sim_top_k: int = 5, hub_edge_cap: int = 200, direct_cap: int = 6, avg_degree: float = 6.0,
                        node_cap: int = 10, hub_cap: int = 60, fr_iter: int = 120, seed: int = 0) -> dict:
    """Returns {"nodes": node indices, "edges": [(src, dst, type, weight)], "x", "y"}."""
    rng = np.random.default_rng(seed)
    edges = []
    # (a) strongest cluster-graph edges (co-following similarity / weighted relations) among labelled accounts
    S = _topk_per_row(W_cluster, sim_top_k)
    S = S.maximum(S.T).tocoo()
    for i, j, w in zip(S.row, S.col, S.data):
        if i < j:
            edges.append((int(lab[i]), int(lab[j]), "similar", float(w)))
    # (b) direct follow edges among labelled accounts, capped per node (mutual first)
    hubs = np.zeros(0, int)
    if "follow" in gd.relations:
        A = gd.relations["follow"].tocsr()
        sub = (A[lab][:, lab] > 0).astype(np.float32).tocsr()
        mutual = sub.minimum(sub.T.tocsr()).tocsr()
        one_way = (sub - mutual).tocsr()
        one_way.eliminate_zeros()
        for M, ty, w in ((mutual, "mutual", 1.0), (one_way, "follow", 0.5)):
            M = M.tocsr()
            for i in range(M.shape[0]):
                js = M.indices[M.indptr[i]:M.indptr[i + 1]]
                if ty == "mutual":
                    js = js[js > i]
                if len(js) > direct_cap:
                    js = rng.choice(js, direct_cap, replace=False)
                for j in js:
                    edges.append((int(lab[i]), int(lab[j]), ty, w))
        # (c) external hubs: most followed by labelled accounts
        ext = np.flatnonzero(gd.y < 0)
        if len(ext) and hub_count > 0:
            indeg_lab = np.asarray((A[lab] > 0).sum(axis=0)).ravel()
            indeg_lab[lab] = 0
            hubs = np.argsort(-indeg_lab)[:hub_count]
            hubs = hubs[indeg_lab[hubs] >= 2]
            AT = (A[:, hubs] > 0).tocsc()
            lab_set = set(lab.tolist())
            for k, h in enumerate(hubs):
                followers = AT.indices[AT.indptr[k]:AT.indptr[k + 1]]
                followers = np.array([u for u in followers if u in lab_set], dtype=int)
                if len(followers) > hub_edge_cap:
                    followers = rng.choice(followers, hub_edge_cap, replace=False)
                for u in followers:
                    edges.append((int(u), int(h), "follow_hub", 0.3))
    # de-duplicate (keep the strongest type per pair)
    best = {}
    for s_, t_, ty, w in edges:
        if s_ == t_:
            continue
        key = (min(s_, t_), max(s_, t_))
        if key not in best or w > best[key][3]:
            best[key] = (s_, t_, ty, w)
    edges = list(best.values())

    nodes = np.concatenate([lab, hubs]).astype(int)
    pos = {int(v): i for i, v in enumerate(nodes)}
    is_hub = np.zeros(len(nodes), bool)
    is_hub[len(lab):] = True
    n_cand = len(edges)
    edges = select_edges(edges, len(nodes), pos, is_hub, rng, avg_degree=avg_degree, node_cap=node_cap, hub_cap=hub_cap)
    e_idx = [(pos[s_], pos[t_]) for s_, t_, _, _ in edges]
    weights = [max(w, 0.05) * (2.0 if ty == "similar" else 1.0) for _, _, ty, w in edges]
    log.info("display graph: %d nodes, %d edges kept of %d candidates; computing layout (DrL + %d FR iterations)",
             len(nodes), len(edges), n_cand, fr_iter)
    xy, n_iso = layout_positions(len(nodes), e_idx, weights, rng, fr_iter=fr_iter)
    log.info("display layout done: %d isolated nodes on the outer ring", n_iso)
    return {"nodes": nodes, "edges": edges, "x": xy[:, 0], "y": xy[:, 1]}
