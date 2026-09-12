import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from trustscore.graph import row_normalize
from trustscore.propagation import build_prior, explain_node, propagate, trust_from_residual


def planted_graph(n_honest=400, n_sybil=200, p_in=0.05, n_attack=60, seed=0):
    rng = np.random.default_rng(seed)
    n = n_honest + n_sybil
    A = sp.lil_matrix((n, n))
    for lo, hi in ((0, n_honest), (n_honest, n)):
        m = hi - lo
        mask = rng.random((m, m)) < p_in
        mask = np.triu(mask, 1)
        r, c = np.nonzero(mask)
        for a, b in zip(r + lo, c + lo):
            A[a, b] = 1
            A[b, a] = 1
    for _ in range(n_attack):  # attack edges: sybil -> honest (one-way follows)
        A[rng.integers(n_honest, n), rng.integers(0, n_honest)] = 1
    y = np.r_[np.zeros(n_honest, int), np.ones(n_sybil, int)]
    return A.tocsr(), y


def test_propagation_bounds_and_auc():
    A, y = planted_graph()
    n = len(y)
    P = row_normalize(A.maximum(A.T))
    rng = np.random.default_rng(1)
    seeds = rng.choice(n, size=30, replace=False)
    q = build_prior(n, seeds, y[seeds])
    r, info = propagate(P, q, alpha=0.8, return_info=True)
    assert info["converged"]
    assert np.all(np.abs(r) <= 0.5 + 1e-9)
    T = trust_from_residual(r)
    test = np.setdiff1d(np.arange(n), seeds)
    auc = roc_auc_score(y[test], 1 - T[test])
    assert auc > 0.95, auc


def test_explanation_reconstructs_residual():
    A, y = planted_graph(n_honest=50, n_sybil=30, seed=3)
    n = len(y)
    channels = {"mutual": A.maximum(A.T).tocsr(), "follow_in": A.T.tocsr()}
    weights = {"mutual": 1.0, "follow_in": 0.5}
    from trustscore.graph import combine_channels
    P = combine_channels(channels, weights)
    q = build_prior(n, np.array([0, 1, n - 1]), np.array([0, 0, 1]))
    r = propagate(P, q, alpha=0.7, tol=1e-12)
    ex = explain_node(5, channels, weights, q, r, alpha=0.7)
    assert abs(ex["prior_term"] + ex["neighbor_term"] - r[5]) < 1e-9


def test_alpha_zero_is_prior():
    n = 10
    P = row_normalize(sp.csr_matrix(np.ones((n, n)) - np.eye(n)))
    q = np.linspace(-0.5, 0.5, n)
    r = propagate(P, q, alpha=0.0)
    assert np.allclose(r, q)
