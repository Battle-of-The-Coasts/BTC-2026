"""Local (node-level) bot classifier: Random Forest on profile + structural features (Botometer-style prior).

Also provides per-prediction feature contributions (Saabas / treeinterpreter decomposition):
    p(x) = bias + sum_f contribution_f(x)
so the UI can say *why* an account looks suspicious before any propagation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from .config import LocalModelConfig


class LocalModel:
    def __init__(self, cfg: LocalModelConfig | None = None, random_state: int | None = None):
        self.cfg = cfg or LocalModelConfig()
        self.rf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators, min_samples_leaf=self.cfg.min_samples_leaf,
            max_features=self.cfg.max_features, class_weight=self.cfg.class_weight,
            random_state=self.cfg.random_state if random_state is None else random_state, n_jobs=-1,
        )
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LocalModel":
        self.feature_names = list(X.columns)
        self.rf.fit(X.to_numpy(np.float32), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(bot)."""
        proba = self.rf.predict_proba(X.to_numpy(np.float32))
        cls = list(self.rf.classes_)
        return proba[:, cls.index(1)] if 1 in cls else np.zeros(len(X))

    def importances(self, X: pd.DataFrame | None = None, y: np.ndarray | None = None, n_repeats: int = 5) -> dict:
        out = {"impurity": dict(zip(self.feature_names, map(float, self.rf.feature_importances_)))}
        if X is not None and y is not None and len(X) > 20:
            pi = permutation_importance(self.rf, X.to_numpy(np.float32), y, scoring="roc_auc", n_repeats=n_repeats,
                                        random_state=0, n_jobs=-1)
            out["permutation_auc"] = dict(zip(self.feature_names, map(float, pi.importances_mean)))
        return out

    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Saabas decomposition. Returns (bias, C) with C of shape (n, d): P(bot) = bias + C.sum(1)."""
        Xn = X.to_numpy(np.float32)
        n, d = Xn.shape
        bot = list(self.rf.classes_).index(1)
        C = np.zeros((n, d))
        bias = 0.0
        for est in self.rf.estimators_:
            t = est.tree_
            val = t.value[:, 0, :]
            val = val / np.maximum(val.sum(axis=1, keepdims=True), 1e-12)
            pv = val[:, bot]
            parent = np.full(t.node_count, -1)
            for j in range(t.node_count):
                for child in (t.children_left[j], t.children_right[j]):
                    if child >= 0:
                        parent[child] = j
            nonroot = np.flatnonzero(parent >= 0)
            delta = pv[nonroot] - pv[parent[nonroot]]
            feat = t.feature[parent[nonroot]]
            F = sp.csr_matrix((delta, (nonroot, feat)), shape=(t.node_count, d))
            paths = est.decision_path(Xn)
            C += (paths @ F).toarray()
            bias += pv[0]
        k = len(self.rf.estimators_)
        return bias / k, C / k
