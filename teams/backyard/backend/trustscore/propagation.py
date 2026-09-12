"""Trust propagation solvers.

Main formula (damped SybilSCAR-D / regularised label propagation):

    r = (1 - alpha) * q + alpha * P r        =>   r* = (1 - alpha) (I - alpha P)^{-1} q
    T = 1/2 - r*                              (trust score in [0, 1]; 1 = fully trusted)

* q  : prior residual of "being a bot"  (+0.5 labelled bot seed, -0.5 labelled human seed,
       lam * (p_local - 0.5) for nodes with a local-model probability, 0 otherwise)
* P  : row-stochastic gather matrix built from typed, hub-damped edge channels (see graph.py)
* alpha in [0, 1): fraction of a node's residual explained by its neighbourhood.
Because P is row-stochastic and |q| <= 1/2, the fixed point satisfies |r*| <= 1/2, so T is in [0, 1].
Jacobi iteration converges geometrically with rate alpha (spectral radius of alpha*P <= alpha).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .graph import combine_channels, row_normalize


def build_prior(n: int, seed_idx: np.ndarray, seed_y: np.ndarray, p_local: np.ndarray | None = None,
                lam: float = 0.5, seed_residual: float = 0.5) -> np.ndarray:
    """Prior residual vector q (positive = evidence of being a bot)."""
    q = np.zeros(n, dtype=np.float64)
    if p_local is not None:
        pl = np.asarray(p_local, dtype=np.float64)
        has = np.isfinite(pl)
        q[has] = lam * (np.clip(pl[has], 0.0, 1.0) - 0.5)
    seed_idx = np.asarray(seed_idx)
    if len(seed_idx):
        q[seed_idx] = np.where(np.asarray(seed_y) == 1, seed_residual, -seed_residual)
    return q


def propagate(P: sp.csr_matrix, q: np.ndarray, alpha: float = 0.8, max_iter: int = 300, tol: float = 1e-8,
              return_info: bool = False):
    """Solve r = (1-alpha) q + alpha P r by Jacobi iteration (P row-stochastic)."""
    if alpha <= 0:
        return (q.copy(), {"iterations": 0, "converged": True, "residual": 0.0}) if return_info else q.copy()
    r = q.copy()
    base = (1.0 - alpha) * q
    it, err = 0, np.inf
    for it in range(1, max_iter + 1):
        r_new = base + alpha * (P @ r)
        err = float(np.max(np.abs(r_new - r)))
        r = r_new
        if err < tol:
            break
    info = {"iterations": it, "converged": bool(err < tol), "residual": err}
    return (r, info) if return_info else r


def trust_from_residual(r: np.ndarray) -> np.ndarray:
    return np.clip(0.5 - r, 0.0, 1.0)


def trust_score(channels: dict[str, sp.csr_matrix], weights: dict[str, float], q: np.ndarray, alpha: float,
                max_iter: int = 300, tol: float = 1e-8, return_info: bool = False):
    P = combine_channels(channels, weights)
    out = propagate(P, q, alpha=alpha, max_iter=max_iter, tol=tol, return_info=return_info)
    if return_info:
        r, info = out
        return trust_from_residual(r), r, info
    return trust_from_residual(out), out


def explain_node(v: int, channels: dict[str, sp.csr_matrix], weights: dict[str, float], q: np.ndarray,
                 r: np.ndarray, alpha: float, top_k: int = 15) -> dict:
    """Decompose r_v = (1-alpha) q_v + alpha * sum_c w_c sum_u G_c[v,u]/D_v r_u into prior, channel and neighbour terms."""
    rows, D = {}, 0.0
    for name, G in channels.items():
        row = G.getrow(v)
        if row.nnz == 0:
            continue
        D += float(row.data.sum())
        rows[name] = (row.indices, row.data)
    prior_term = (1.0 - alpha) * float(q[v])
    per_channel, per_neighbor = {}, []
    if D > 0:
        for name, (idx, g) in rows.items():
            w = float(weights.get(name, 0.0))
            contrib = alpha * w * g / D * r[idx]
            per_channel[name] = {"contribution": float(contrib.sum()), "n_neighbors": int(len(idx)),
                                 "weight_share": float(g.sum() / D), "homophily_weight": w}
            for u, c, wt in zip(idx, contrib, w * g / D):
                per_neighbor.append((int(u), float(c), float(wt), name))
    per_neighbor.sort(key=lambda t: -abs(t[1]))
    neighbor_term = sum(c for _, c, _, _ in per_neighbor)
    return {
        "node": int(v), "residual": float(r[v]), "trust": float(np.clip(0.5 - r[v], 0, 1)),
        "prior_residual": float(q[v]), "prior_term": prior_term, "neighbor_term": float(neighbor_term),
        "n_neighbors": int(len(per_neighbor)),
        "per_channel": per_channel,
        "top_neighbors": [{"node": u, "contribution": c, "weight": wt, "channel": ch} for u, c, wt, ch in per_neighbor[:top_k]],
    }


# ------------------------------------------------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------------------------------------------------
def personalized_pagerank(A_walk: sp.csr_matrix, seed_idx: np.ndarray, alpha: float = 0.85, max_iter: int = 200,
                          tol: float = 1e-10) -> np.ndarray:
    """PPR with teleport set = seeds, walking along A_walk (u -> v). Column-stochastic transition; dangling mass
    is redistributed to the seeds. Returns a probability vector."""
    n = A_walk.shape[0]
    M = row_normalize(A_walk).T.tocsr()   # M[v,u] = A[u,v]/outdeg(u)
    out_zero = np.asarray(A_walk.sum(axis=1)).ravel() == 0
    e = np.zeros(n)
    if len(seed_idx) == 0:
        return e
    e[np.asarray(seed_idx)] = 1.0 / len(seed_idx)
    t = e.copy()
    for _ in range(max_iter):
        dangling = float(t[out_zero].sum())
        t_new = (1 - alpha) * e + alpha * (M @ t + dangling * e)
        if np.abs(t_new - t).max() < tol:
            t = t_new
            break
        t = t_new
    return t


def trustrank_score(A_follow: sp.csr_matrix, human_seeds: np.ndarray, bot_seeds: np.ndarray, alpha: float = 0.85,
                    degree_normalize: bool = True) -> np.ndarray:
    """TrustRank/SybilRank-style baseline: trust walks follower -> followee from human seeds, distrust from bot seeds
    (Anti-TrustRank walks the reverse direction: who follows a bot?). Score = trust - distrust, in (-1, 1) scale."""
    t = personalized_pagerank(A_follow, human_seeds, alpha)
    d = personalized_pagerank(A_follow.T.tocsr(), bot_seeds, alpha)
    if degree_normalize:  # SybilRank: divide by degree so hubs do not dominate
        deg = np.asarray((A_follow > 0).sum(axis=0)).ravel() + np.asarray((A_follow > 0).sum(axis=1)).ravel()
        deg = np.maximum(deg, 1)
        t, d = t / deg, d / deg
    t = t / max(t.max(), 1e-15)
    d = d / max(d.max(), 1e-15)
    return t - d


def sgc_features(A_sym: sp.csr_matrix, X: np.ndarray, k: int = 2) -> np.ndarray:
    """Simplified Graph Convolution (Wu et al. 2019): S^k X with S = D^-1/2 (A + I) D^-1/2."""
    n = A_sym.shape[0]
    A_hat = (A_sym > 0).astype(np.float64) + sp.identity(n, format="csr")
    d = np.asarray(A_hat.sum(axis=1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = (Dm @ A_hat @ Dm).tocsr()
    Z = np.asarray(X, dtype=np.float64)
    for _ in range(k):
        Z = S @ Z
    return Z


# ------------------------------------------------------------------------------------------------------------
# Data-driven channel weights (homophily estimated from seed-seed edges, blended with the hand-set prior)
# ------------------------------------------------------------------------------------------------------------
def estimate_channel_weights(channels: dict[str, sp.csr_matrix], seed_idx: np.ndarray, seed_y: np.ndarray,
                             prior_weights: dict[str, float], shrink: float = 50.0) -> tuple[dict[str, float], dict]:
    """Per-channel homophily strength = phi coefficient (Pearson correlation of the two endpoint labels) measured on
    edges whose endpoints are both seeds, clipped at 0 and blended with the hand-set prior:
        w_c = (n_c * max(phi_c, 0) + shrink * prior_c) / (n_c + shrink).
    With balanced classes phi equals 2 P(same label) - 1, i.e. SybilSCAR's residual homophily (2 w - 1); unlike the
    raw same-label rate it is not inflated by the majority class, so a channel whose edges all point at humans
    regardless of the sender's label gets phi ~ 0 and transmits nothing."""
    seed_idx = np.asarray(seed_idx)
    n = next(iter(channels.values())).shape[0]
    lab = np.full(n, -1, dtype=np.int8)
    lab[seed_idx] = np.asarray(seed_y)
    weights, report = {}, {}
    for name, G in channels.items():
        coo = G.tocoo()
        m = (lab[coo.row] >= 0) & (lab[coo.col] >= 0)
        n_c = int(m.sum())
        pattern_symmetric = ((G != 0) != (G.T != 0)).nnz == 0
        if pattern_symmetric:
            n_c //= 2   # undirected channel: each seed-seed edge appears twice in the sparse matrix
        a = lab[coo.row][m].astype(float)   # receiver label
        b = lab[coo.col][m].astype(float)   # sender label
        same = float((a == b).mean()) if n_c else float("nan")
        prior = float(prior_weights.get(name, 0.0))
        if n_c >= 2 and a.std() > 0 and b.std() > 0:
            phi = float(np.corrcoef(a, b)[0, 1])
            n_eff = n_c
        else:
            # phi is undefined when one class never appears at one end of the channel's seed edges (e.g. every
            # mutual follow between seeds is human-human). No evidence either way -> keep the hand prior.
            phi, n_eff = float("nan"), 0
        w = (n_eff * max(0.0 if np.isnan(phi) else phi, 0.0) + shrink * prior) / (n_eff + shrink)
        weights[name] = float(np.clip(w, 0.0, 1.0))
        report[name] = {"seed_edges": n_c, "same_label_rate": None if np.isnan(same) else round(same, 4),
                        "phi": None if np.isnan(phi) else round(phi, 4), "prior": prior, "weight": round(weights[name], 4)}
    p = float((np.asarray(seed_y) == 1).mean()) if len(seed_y) else 0.5
    report["_independence_same_rate"] = round(p * p + (1 - p) * (1 - p), 4)
    return weights, report
