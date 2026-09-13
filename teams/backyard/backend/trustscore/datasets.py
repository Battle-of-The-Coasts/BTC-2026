"""Dataset loaders producing one canonical in-memory representation (:class:`GraphData`).

Supported open datasets
-----------------------
* ``cresci-2015`` (Cresci et al. 2015, "Fame for sale"; Bot Repository release): 5,301 labelled Twitter accounts
  (1,950 genuine from TFP/E13, 3,351 fake followers bought from FSF/INT/TWT) with raw profiles and the full
  follower/friend lists of every labelled account (4.25M follow edges, 1.29M external accounts).
* ``mgtab`` (Shi et al. 2023): 10,199 expert-annotated accounts, 7 relation types, 788 pre-extracted features.

Edge convention: for directed relations ``A[u, v] > 0`` means "u -> v" (u follows / mentions / replies to / quotes v).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import DataConfig
from .ptload import load_pt

log = logging.getLogger(__name__)

CRESCI_SUBSETS = {"E13": 0, "TFP": 0, "FSF": 1, "INT": 1, "TWT": 1}  # 0 = human, 1 = bot (fake follower)
MGTAB_RELATIONS = ["followers", "friends", "mention", "reply", "quoted", "url", "hashtag"]
MGTAB_PROFILE_DIMS = 20  # first 20 columns = user property features (paper: "20 user property features"), rest = text embeddings


@dataclass
class GraphData:
    name: str
    node_ids: np.ndarray                      # (N,) original identifiers (int64 for Twitter ids, int for MGTAB)
    y: np.ndarray                             # (N,) int8: 1 = bot / untrusted, 0 = human, -1 = unknown
    subset: np.ndarray                        # (N,) str: collection subset ('' for external accounts)
    relations: dict[str, sp.csr_matrix]       # relation name -> N x N sparse matrix (see edge convention above)
    symmetric_relations: set[str] = field(default_factory=set)
    profile: pd.DataFrame | None = None       # raw profile fields, index = node index (only nodes that have a profile)
    X_pre: np.ndarray | None = None           # pre-extracted numeric feature block aligned to node index (NaN if missing)
    X_pre_names: list[str] = field(default_factory=list)
    X_content: np.ndarray | None = None       # optional content-embedding block (MGTAB), aligned to node index
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.node_ids)

    @property
    def labeled_idx(self) -> np.ndarray:
        return np.flatnonzero(self.y >= 0)

    def summary(self) -> dict:
        lab = self.labeled_idx
        out = {
            "name": self.name, "n_nodes": int(self.n), "n_labeled": int(len(lab)),
            "n_bots": int((self.y == 1).sum()), "n_humans": int((self.y == 0).sum()),
            "relations": {k: int(v.nnz) for k, v in self.relations.items()},
        }
        out.update(self.meta)
        return out


# --------------------------------------------------------------------------------------------------------------
# cresci-2015
# --------------------------------------------------------------------------------------------------------------
def load_cresci2015(raw_dir: str, cfg: DataConfig | None = None) -> GraphData:
    cfg = cfg or DataConfig()
    base = os.path.join(raw_dir, "cresci-2015")
    users, edges = [], []
    for s in CRESCI_SUBSETS:
        u = pd.read_csv(os.path.join(base, s, "users.csv"), dtype={"id": "int64"}, low_memory=False)
        u["subset"] = s
        users.append(u)
        # followers.csv: (follower, account) ; friends.csv: (account, followee) -> both are "source follows target"
        fo = pd.read_csv(os.path.join(base, s, "followers.csv"), dtype="int64")
        fr = pd.read_csv(os.path.join(base, s, "friends.csv"), dtype="int64")
        edges += [fo, fr]
    users = pd.concat(users, ignore_index=True)
    assert users["id"].is_unique
    edges = pd.concat(edges, ignore_index=True).drop_duplicates()
    edges = edges[edges.source_id != edges.target_id]
    n_edges_raw = len(edges)

    labeled_ids = users["id"].to_numpy()
    lab_set = pd.Index(labeled_ids)
    src_l = edges.source_id.isin(lab_set).to_numpy()
    dst_l = edges.target_id.isin(lab_set).to_numpy()
    ext = pd.concat([edges.source_id[~src_l], edges.target_id[~dst_l]])
    ext_deg = ext.value_counts()
    n_ext_raw = len(ext_deg)
    keep_ext = ext_deg.index[ext_deg.to_numpy() >= cfg.min_external_degree].to_numpy()
    keep_ext.sort()
    keep_set = pd.Index(keep_ext)
    ok = (src_l | edges.source_id.isin(keep_set).to_numpy()) & (dst_l | edges.target_id.isin(keep_set).to_numpy())
    edges = edges[ok]

    node_ids = np.concatenate([labeled_ids, keep_ext])
    index = pd.Series(np.arange(len(node_ids)), index=node_ids)
    src = index.loc[edges.source_id.to_numpy()].to_numpy()
    dst = index.loc[edges.target_id.to_numpy()].to_numpy()
    N = len(node_ids)
    A = sp.csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(N, N))
    A.sum_duplicates()
    A.data[:] = 1.0

    y = np.full(N, -1, dtype=np.int8)
    y[: len(labeled_ids)] = users["subset"].map(CRESCI_SUBSETS).to_numpy()
    subset = np.array([""] * N, dtype=object)
    subset[: len(labeled_ids)] = users["subset"].to_numpy()
    profile = users.drop(columns=["dataset"], errors="ignore").copy()
    profile.index = np.arange(len(labeled_ids))

    meta = {
        "n_edges_raw": int(n_edges_raw), "n_external_raw": int(n_ext_raw),
        "n_external_kept": int(len(keep_ext)), "n_edges_kept": int(A.nnz),
        "min_external_degree": cfg.min_external_degree,
        "label_semantics": "1 = purchased fake follower (FSF/INT/TWT), 0 = verified genuine account (TFP/E13)",
    }
    log.info("cresci-2015: %d nodes (%d labelled), %d follow edges", N, len(labeled_ids), A.nnz)
    return GraphData(name="cresci-2015", node_ids=node_ids, y=y, subset=subset,
                     relations={"follow": A}, profile=profile, meta=meta)


# --------------------------------------------------------------------------------------------------------------
# MGTAB
# --------------------------------------------------------------------------------------------------------------
def _edge_overlap(a: sp.csr_matrix, b: sp.csr_matrix) -> float:
    inter = a.multiply(b).nnz
    return inter / max(1, min(a.nnz, b.nnz))


def load_mgtab(raw_dir: str, cfg: DataConfig | None = None) -> GraphData:
    base = os.path.join(raw_dir, "MGTAB", "MGTAB")
    X = load_pt(os.path.join(base, "features.pt")).astype(np.float32)
    y = load_pt(os.path.join(base, "labels_bot.pt")).astype(np.int8)
    ei = load_pt(os.path.join(base, "edge_index.pt"))
    et = load_pt(os.path.join(base, "edge_type.pt"))
    ew = load_pt(os.path.join(base, "edge_weight.pt")).astype(np.float32)
    N = X.shape[0]

    def mat(mask):
        m = sp.csr_matrix((ew[mask], (ei[0, mask], ei[1, mask])), shape=(N, N))
        m.sum_duplicates()
        return m

    rel = {name: mat(et == i) for i, name in enumerate(MGTAB_RELATIONS)}
    # Direction convention of the release is undocumented. Both 'followers' and 'friends' describe the same
    # relation seen from the two sides, so the convention under which they overlap most is the consistent one.
    same = _edge_overlap(rel["followers"], rel["friends"])
    reversed_ = _edge_overlap(rel["followers"].T.tocsr(), rel["friends"])
    if reversed_ >= same:
        follow = rel["friends"].maximum(rel["followers"].T.tocsr())
        convention = "followers edges reversed (u->v in 'followers' means v follows u)"
    else:
        follow = rel["friends"].maximum(rel["followers"])
        convention = "followers edges kept as-is"
    follow.setdiag(0)
    follow.eliminate_zeros()
    relations = {"follow": follow.tocsr()}
    for name in ("mention", "reply", "quoted"):
        m = rel[name].tocsr()
        m.setdiag(0)
        m.eliminate_zeros()
        m.data = np.log1p(m.data)   # repeated (u, v) interactions are summed by sum_duplicates(); damp the counts
        relations[name] = m
    sym = set()
    for name in ("url", "hashtag"):
        m = rel[name].maximum(rel[name].T).tocsr()
        m.setdiag(0)
        m.eliminate_zeros()
        relations[name] = m
        sym.add(name)

    names = [f"prop_{i}" for i in range(MGTAB_PROFILE_DIMS)]
    meta = {
        "follow_direction_convention": convention,
        "follow_overlap_same": round(float(same), 4), "follow_overlap_reversed": round(float(reversed_), 4),
        "profile_feature_dims": MGTAB_PROFILE_DIMS, "content_feature_dims": int(X.shape[1] - MGTAB_PROFILE_DIMS),
        "label_semantics": "1 = bot, 0 = human (expert annotation, MGTAB)",
    }
    log.info("MGTAB: %d nodes, relations %s (%s)", N, {k: v.nnz for k, v in relations.items()}, convention)
    return GraphData(name="mgtab", node_ids=np.arange(N), y=y, subset=np.array(["mgtab"] * N, dtype=object),
                     relations=relations, symmetric_relations=sym, profile=None,
                     X_pre=X[:, :MGTAB_PROFILE_DIMS], X_pre_names=names, X_content=X[:, MGTAB_PROFILE_DIMS:], meta=meta)


LOADERS = {"cresci-2015": load_cresci2015, "mgtab": load_mgtab}


def load_dataset(name: str, raw_dir: str, cfg: DataConfig | None = None) -> GraphData:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; available: {list(LOADERS)}")
    return LOADERS[name](raw_dir, cfg)
