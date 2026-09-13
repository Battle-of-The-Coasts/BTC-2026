"""Cluster, coordination and anomaly detection on the account graph.

* co-following similarity graph (TF-IDF-weighted cosine over followee sets) -> Leiden communities
* dense-block detection (Fraudar-style greedy peeling, Bahmani rounds approximation) = lockstep co-following groups
* creation bursts (many accounts created the same day)
The detectors are unsupervised; labels are only used afterwards to *report* how pure each cluster is.
"""
from __future__ import annotations

import logging

import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import ClusterConfig
from .datasets import GraphData
from .graph import degrees

log = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------------------
# similarity graph
# ------------------------------------------------------------------------------------------------------------
def cofollow_similarity(A: sp.csr_matrix, rows: np.ndarray, top_k: int = 10, min_sim: float = 0.05,
                        chunk: int = 512) -> sp.csr_matrix:
    """Cosine similarity between the followee sets of `rows` (TF-IDF weighted: popular followees count less).
    Returns a symmetric sparse (len(rows) x len(rows)) matrix keeping the top_k most similar accounts per row."""
    B = (A[rows] > 0).astype(np.float32).tocsr()
    df = np.asarray(B.sum(axis=0)).ravel()
    idf = np.log((len(rows) + 1) / (df + 1)) + 1.0
    Bw = (B @ sp.diags(idf.astype(np.float32))).tocsr()
    norms = np.sqrt(np.asarray(Bw.multiply(Bw).sum(axis=1)).ravel())
    Bw = (sp.diags(np.where(norms > 0, 1.0 / np.maximum(norms, 1e-12), 0.0).astype(np.float32)) @ Bw).tocsr()
    BwT = Bw.T.tocsc()
    n = len(rows)
    rows_out, cols_out, vals_out = [], [], []
    for start in range(0, n, chunk):
        S = (Bw[start:start + chunk] @ BwT).tocsr()
        for i in range(S.shape[0]):
            lo, hi = S.indptr[i], S.indptr[i + 1]
            idx, val = S.indices[lo:hi], S.data[lo:hi]
            m = (idx != start + i) & (val >= min_sim)
            idx, val = idx[m], val[m]
            if len(val) > top_k:
                keep = np.argpartition(-val, top_k)[:top_k]
                idx, val = idx[keep], val[keep]
            rows_out.append(np.full(len(idx), start + i)); cols_out.append(idx); vals_out.append(val)
    if not rows_out:
        return sp.csr_matrix((n, n), dtype=np.float32)
    S = sp.csr_matrix((np.concatenate(vals_out), (np.concatenate(rows_out), np.concatenate(cols_out))), shape=(n, n))
    return S.maximum(S.T).tocsr()


def cluster_graph(gd: GraphData, cfg: ClusterConfig, relation_weights: dict[str, float]) -> tuple[sp.csr_matrix, np.ndarray, str]:
    """Weighted symmetric graph over labelled accounts used for community detection.
    cresci-2015: co-following similarity (top-k) + direct mutual follows. MGTAB: weighted union of its relations."""
    lab = gd.labeled_idx
    if gd.name == "cresci-2015":
        A = gd.relations["follow"].tocsr()
        S = cofollow_similarity(A, lab, top_k=cfg.similarity_top_k, min_sim=cfg.similarity_min)
        sub = A[lab][:, lab]
        mutual = sub.minimum(sub.T.tocsr())
        mutual.data[:] = 1.0
        W = S.maximum(mutual.astype(np.float32)).tocsr()
        mode = "co-following similarity (top-%d, TF-IDF cosine >= %.2f) + mutual follows" % (cfg.similarity_top_k, cfg.similarity_min)
    else:
        acc = None
        for name, A in gd.relations.items():
            sub = (A[lab][:, lab] > 0).astype(np.float32)
            if name == "follow":
                mutual = sub.minimum(sub.T.tocsr())
                one = (sub - mutual)
                one = one.maximum(one.T)
                part = mutual * relation_weights.get("mutual", 1.0) + one * relation_weights.get("follow_in", 0.5)
            elif name in gd.symmetric_relations:
                part = sub * relation_weights.get(name, 0.25)
            else:
                part = sub.maximum(sub.T) * relation_weights.get(f"{name}_in", 0.5)
            acc = part if acc is None else acc + part
        W = acc.tocsr()
        W.setdiag(0)
        W.eliminate_zeros()
        mode = "weighted union of relations restricted to labelled accounts"
    return W, lab, mode


def leiden_communities(W: sp.csr_matrix, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    coo = sp.triu(W, k=1).tocoo()
    g = ig.Graph(n=W.shape[0], edges=np.column_stack([coo.row, coo.col]), directed=False)
    g.es["weight"] = coo.data.astype(float)
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=seed)
    return np.asarray(part.membership)


# ------------------------------------------------------------------------------------------------------------
# dense-block detection (Fraudar-style)
# ------------------------------------------------------------------------------------------------------------
def densest_block(M: sp.csr_matrix, eps: float = 0.1, min_rows: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
    """Approximate densest bipartite block of M (rows x cols, weighted) by peeling in rounds (Bahmani et al. 2012):
    repeatedly delete every node whose weighted degree is <= (1+eps) * current average degree; return the best
    density (total weight / (#rows + #cols)) seen. Columns are weighted 1/log(deg+5) beforehand (Fraudar) so that
    very popular targets cannot dominate."""
    M = M.tocsr().astype(np.float64)
    col_deg = np.asarray((M > 0).sum(axis=0)).ravel()
    Mw = (M @ sp.diags(1.0 / np.log(col_deg + 5.0))).tocsr()
    MwT = Mw.T.tocsr()
    r_alive = np.ones(Mw.shape[0], bool)
    c_alive = np.ones(Mw.shape[1], bool)
    best = (None, None, -1.0)
    while r_alive.sum() >= min_rows and c_alive.any():
        rdeg = Mw @ c_alive.astype(float)
        cdeg = MwT @ r_alive.astype(float)
        rdeg[~r_alive] = 0
        cdeg[~c_alive] = 0
        total = rdeg.sum()
        size = r_alive.sum() + c_alive.sum()
        dens = total / size
        if dens > best[2]:
            best = (np.flatnonzero(r_alive), np.flatnonzero(c_alive), float(dens))
        thr = (1 + eps) * (2 * total / size)   # 2*avg because each edge counts once per side
        kill_r = r_alive & (rdeg <= thr)
        kill_c = c_alive & (cdeg <= thr)
        if not kill_r.any() and not kill_c.any():
            break
        r_alive &= ~kill_r
        c_alive &= ~kill_c
    return best


def dense_blocks(gd: GraphData, n_blocks: int = 3, min_rows: int = 5) -> list[dict]:
    """Peel off the n densest lockstep blocks of the labelled-account -> target follow matrix."""
    lab = gd.labeled_idx
    if "follow" not in gd.relations:
        return []
    A = gd.relations["follow"].tocsr()
    M = (A[lab] > 0).astype(np.float32).tocsr()
    col_deg = np.asarray(M.sum(axis=0)).ravel()
    keep_cols = np.flatnonzero(col_deg >= 2)
    M = M[:, keep_cols].tocsr()
    blocks = []
    alive_rows = np.ones(len(lab), bool)
    for b in range(n_blocks):
        rows_idx = np.flatnonzero(alive_rows)
        if len(rows_idx) < min_rows:
            break
        r, c, dens = densest_block(M[rows_idx], min_rows=min_rows)
        if r is None or len(r) < min_rows:
            break
        members = lab[rows_idx[r]]
        targets = keep_cols[c]
        blocks.append({"id": b, "members": members, "targets": targets, "density": dens})
        alive_rows[rows_idx[r]] = False
    return blocks


# ------------------------------------------------------------------------------------------------------------
# creation bursts
# ------------------------------------------------------------------------------------------------------------
def creation_bursts(profile: pd.DataFrame | None, min_accounts: int = 20) -> tuple[np.ndarray, list[dict]]:
    """Flag accounts created on a day when >= min_accounts dataset accounts were created."""
    if profile is None or "created_at" not in profile:
        return np.zeros(0, bool), []
    created = pd.to_datetime(profile["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    day = created.dt.floor("D")
    counts = day.value_counts()
    burst_days = counts[counts >= min_accounts]
    flag = day.isin(burst_days.index).to_numpy()
    days = [{"day": str(d.date()), "n_accounts": int(n)} for d, n in burst_days.sort_values(ascending=False).items()]
    return flag, days


# ------------------------------------------------------------------------------------------------------------
# reporting
# ------------------------------------------------------------------------------------------------------------
def community_report(gd: GraphData, W: sp.csr_matrix, lab: np.ndarray, membership: np.ndarray, trust: np.ndarray,
                     p_bot: np.ndarray, burst_flag: np.ndarray | None = None, top_targets: int = 5) -> list[dict]:
    y = gd.y[lab]
    A = gd.relations["follow"].tocsr() if "follow" in gd.relations else None
    out = []
    for cid in np.unique(membership):
        m = np.flatnonzero(membership == cid)
        nodes = lab[m]
        Wsub = W[m][:, m]
        possible = len(m) * (len(m) - 1) / 2
        density = float(sp.triu(Wsub, 1).nnz / possible) if possible > 0 else 0.0
        rec = {
            "id": int(cid), "size": int(len(m)),
            "mean_trust": float(trust[nodes].mean()), "min_trust": float(trust[nodes].min()),
            "predicted_bot_share": float((p_bot[nodes] > 0.5).mean()),
            "true_bot_share": float((y[m] == 1).mean()),
            "internal_density": density,
            "subsets": {str(k): int(v) for k, v in zip(*np.unique(gd.subset[nodes], return_counts=True))},
            "burst_share": float(burst_flag[m].mean()) if burst_flag is not None and len(burst_flag) else None,
            "members": [int(i) for i in nodes],
        }
        if A is not None and top_targets > 0 and len(m) >= 2:
            sub = A[nodes]
            cnt = np.asarray((sub > 0).sum(axis=0)).ravel()
            top = np.argsort(-cnt)[:top_targets]
            rec["top_targets"] = [{"node": int(t), "followed_by": int(cnt[t]), "share": float(cnt[t] / len(m))}
                                  for t in top if cnt[t] >= 2]
        out.append(rec)
    out.sort(key=lambda r: -r["size"])
    return out
