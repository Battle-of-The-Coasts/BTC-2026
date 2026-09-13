import numpy as np
import pandas as pd
import scipy.sparse as sp

from trustscore.clusters import cofollow_similarity, densest_block, leiden_communities
from trustscore.datasets import GraphData
from trustscore.features import structural_features
from trustscore.graph import build_channels, combine_channels
from trustscore.propagation import estimate_channel_weights


def toy():
    # 6 labelled accounts (0-2 humans, 3-5 bots) + 4 external targets (6-9); bots co-follow 8, 9; humans co-follow 6, 7
    n = 10
    A = sp.lil_matrix((n, n))
    for u in (0, 1, 2):
        A[u, 6] = 1; A[u, 7] = 1
    for u in (3, 4, 5):
        A[u, 8] = 1; A[u, 9] = 1
    A[0, 1] = 1; A[1, 0] = 1   # mutual follow between two humans
    A[3, 0] = 1                # attack edge: bot follows human
    y = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1, -1], dtype=np.int8)
    return GraphData(name="toy", node_ids=np.arange(n), y=y, subset=np.array(["h"] * 3 + ["b"] * 3 + [""] * 4, dtype=object),
                     relations={"follow": A.tocsr()})


def test_channels_and_weights():
    gd = toy()
    ch = build_channels(gd, hub_damping=0.5)
    assert set(ch) == {"mutual", "follow_in", "follow_out"}
    assert ch["mutual"].nnz == 2
    P = combine_channels(ch, {"mutual": 1.0, "follow_in": 0.5, "follow_out": 0.25})
    rs = np.asarray(P.sum(axis=1)).ravel()
    assert np.all(rs <= 1 + 1e-9)
    P1 = combine_channels(ch, {"mutual": 1.0, "follow_in": 1.0, "follow_out": 1.0})
    rs1 = np.asarray(P1.sum(axis=1)).ravel()
    assert np.all((np.abs(rs1 - 1) < 1e-9) | (rs1 == 0))
    w, rep = estimate_channel_weights(ch, np.array([0, 1, 3]), np.array([0, 0, 1]), {"mutual": 1.0, "follow_in": 0.5, "follow_out": 0.25}, shrink=1.0)
    assert 0 <= w["mutual"] <= 1 and rep["mutual"]["seed_edges"] == 1   # one undirected human-human seed edge
    assert rep["mutual"]["phi"] is None and abs(w["mutual"] - 1.0) < 1e-9   # phi undefined -> hand prior kept


def test_structural_and_similarity():
    gd = toy()
    f = structural_features(gd)
    assert f.shape[0] == 10 and f.loc[0, "mutual_deg"] == 1 and f.loc[3, "out_deg"] == 3
    S = cofollow_similarity(gd.relations["follow"], np.arange(6), top_k=5, min_sim=0.01)
    assert S[3, 4] > 0.5 and S[0, 3] < 0.2   # bots similar to bots, not to humans
    memb = leiden_communities(S)
    assert memb[3] == memb[4] == memb[5] and memb[0] != memb[3]


def test_densest_block():
    M = sp.lil_matrix((8, 6))
    for i in range(4):
        for j in range(3):
            M[i, j] = 1          # a 4x3 complete block
    M[5, 4] = 1; M[6, 5] = 1
    r, c, dens = densest_block(M.tocsr(), min_rows=2)
    assert set(r) == {0, 1, 2, 3} and set(c) == {0, 1, 2}
