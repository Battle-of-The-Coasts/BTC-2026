"""Typed sparse-graph utilities: edge channels, sender damping, normalisation, undirected views."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .datasets import GraphData


def row_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    M = M.tocsr()
    s = np.asarray(M.sum(axis=1)).ravel()
    inv = np.where(s > 0, 1.0 / np.maximum(s, 1e-12), 0.0)
    return sp.diags(inv) @ M


def _damp(M: sp.csr_matrix, sender_degree: np.ndarray, gamma: float) -> sp.csr_matrix:
    """Scale column u (sender u) of the gather matrix by 1/deg(u)^gamma."""
    d = np.maximum(sender_degree.astype(np.float64), 1.0)
    return (M @ sp.diags(d ** (-gamma))).tocsr()


def build_channels(gd: GraphData, hub_damping: float = 0.5) -> dict[str, sp.csr_matrix]:
    """Return gather matrices G_c (N x N) with G_c[v, u] = strength with which v collects u's residual over channel c.

    Channels derived from a directed 'follow' relation (u -> v means u follows v):
      mutual      : u and v follow each other (symmetric)
      follow_in   : u follows v, one-way   (v collects from its followers)
      follow_out  : v follows u, one-way   (v collects from its followees)
    Other directed relations r give r_in / r_out the same way; symmetric relations give one channel.
    Senders are damped by 1/deg(u)^hub_damping (deg measured in the channel's own direction).
    """
    ch: dict[str, sp.csr_matrix] = {}
    for name, A in gd.relations.items():
        A = A.tocsr()
        if name in gd.symmetric_relations:
            deg = np.asarray((A > 0).sum(axis=1)).ravel()
            ch[name] = _damp(A, deg, hub_damping)
            continue
        if name == "follow":
            mutual = A.minimum(A.T.tocsr())
            one_way = (A - mutual).tocsr()
            one_way.eliminate_zeros()
            mdeg = np.asarray((mutual > 0).sum(axis=1)).ravel()
            ch["mutual"] = _damp(mutual, mdeg, hub_damping)
            outdeg = np.asarray((one_way > 0).sum(axis=1)).ravel()   # how many accounts u follows (one-way)
            indeg = np.asarray((one_way > 0).sum(axis=0)).ravel()    # how many accounts follow u (one-way)
            ch["follow_in"] = _damp(one_way.T.tocsr(), outdeg, hub_damping)   # v <- its followers u
            ch["follow_out"] = _damp(one_way, indeg, hub_damping)              # v <- its followees u
        else:
            outdeg = np.asarray((A > 0).sum(axis=1)).ravel()
            indeg = np.asarray((A > 0).sum(axis=0)).ravel()
            ch[f"{name}_in"] = _damp(A.T.tocsr(), outdeg, hub_damping)
            ch[f"{name}_out"] = _damp(A, indeg, hub_damping)
    return ch


def combine_channels(channels: dict[str, sp.csr_matrix], weights: dict[str, float]) -> sp.csr_matrix:
    """P = D^-1 sum_c w_c G_c with D = diag(sum_c G_c 1) the *unweighted* gather degree.

    Each neighbour message is scaled by its channel's homophily strength w_c in [0, 1] (SybilSCAR semantics:
    an edge type with no label homophily transmits nothing), and the node averages over all its incident
    edges. Row sums are <= 1, so the damped fixed point stays in [-1/2, 1/2]."""
    acc, deg = None, None
    for name, G in channels.items():
        G = G.tocsr()
        d = np.asarray(G.sum(axis=1)).ravel()
        deg = d if deg is None else deg + d
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        acc = G * w if acc is None else acc + G * w
    if acc is None:
        raise ValueError("no active channel")
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    return (sp.diags(inv) @ acc.tocsr()).tocsr()


def undirected_union(gd: GraphData, relations: list[str] | None = None, binary: bool = True) -> sp.csr_matrix:
    """Symmetric union of relations (binary by default), no self loops."""
    acc = None
    for name, A in gd.relations.items():
        if relations is not None and name not in relations:
            continue
        S = A.maximum(A.T.tocsr())
        acc = S if acc is None else acc.maximum(S)
    acc = acc.tocsr()
    acc.setdiag(0)
    acc.eliminate_zeros()
    if binary:
        acc.data[:] = 1.0
    return acc


def follow_matrix(gd: GraphData) -> sp.csr_matrix:
    if "follow" in gd.relations:
        return gd.relations["follow"].tocsr()
    return undirected_union(gd)


def degrees(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """(out_degree, in_degree) counting non-zero entries."""
    B = A > 0
    return np.asarray(B.sum(axis=1)).ravel(), np.asarray(B.sum(axis=0)).ravel()
