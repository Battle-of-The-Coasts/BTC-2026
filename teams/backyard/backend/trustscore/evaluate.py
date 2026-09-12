"""Evaluation protocol and the TrustScorer (model selection + calibration using only the labelled seeds).

All methods return a *bot score* (higher = more likely bot) for every node; metrics are computed on labelled
nodes that were NOT used as seeds. Nothing in the pipeline ever sees the labels of evaluation nodes.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import Config
from .datasets import GraphData
from .graph import combine_channels, follow_matrix, undirected_union
from .local_model import LocalModel
from .propagation import (build_prior, estimate_channel_weights, propagate, sgc_features, trust_from_residual,
                          trustrank_score)

log = logging.getLogger(__name__)

METHODS = ["local_rf", "propagation_only", "trust_score", "trust_score_fixed_w", "trustrank", "sgc"]
ALPHA_GRID = (0.2, 0.4, 0.6, 0.8)
LAM_GRID = (0.5, 1.0)


def sample_seeds(y_lab: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Stratified sample of labelled positions (at least one of each class)."""
    pos = np.flatnonzero(y_lab == 1)
    neg = np.flatnonzero(y_lab == 0)
    k_pos = max(1, int(round(frac * len(pos))))
    k_neg = max(1, int(round(frac * len(neg))))
    return np.concatenate([rng.choice(pos, k_pos, replace=False), rng.choice(neg, k_neg, replace=False)])


def flip_labels(y_seed: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    y = y_seed.copy()
    k = int(round(noise * len(y)))
    if k > 0:
        idx = rng.choice(len(y), k, replace=False)
        y[idx] = 1 - y[idx]
    return y


def stratified_folds(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    fold = np.empty(len(y), int)
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


NO_THRESHOLD = {"trustrank"}   # ranking-only scores: threshold metrics are meaningless and reported as None


def metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5, thresholded: bool = True) -> dict:
    """AUC/AP on the score; F1/precision/recall/accuracy at `threshold` (only for probabilistic scores)."""
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    two = len(np.unique(y_true)) > 1
    out = {"n": int(len(y_true)), "auc": float(roc_auc_score(y_true, score)) if two else None,
           "ap": float(average_precision_score(y_true, score)) if two else None}
    if thresholded:
        pred = (score > threshold).astype(int)
        out.update({"f1": float(f1_score(y_true, pred, zero_division=0)), "precision": float(precision_score(y_true, pred, zero_division=0)),
                    "recall": float(recall_score(y_true, pred, zero_division=0)), "accuracy": float(accuracy_score(y_true, pred))})
    else:
        out.update({"f1": None, "precision": None, "recall": None, "accuracy": None})
    return out


class Evaluator:
    """Holds everything that is independent of the seed split (features, channels, SGC features)."""

    def __init__(self, gd: GraphData, channels: dict[str, sp.csr_matrix], X_local: pd.DataFrame, X_all: pd.DataFrame, cfg: Config):
        self.gd, self.channels, self.X_local, self.X_all, self.cfg = gd, channels, X_local, X_all, cfg
        self.lab = gd.labeled_idx
        self.local_pos = pd.Series(np.arange(len(X_local)), index=X_local.index)
        self.A_follow = follow_matrix(gd)
        t = time.time()
        U = undirected_union(gd)
        self.Z_sgc = sgc_features(U, StandardScaler().fit_transform(X_all.to_numpy(np.float32)), k=2)
        log.info("SGC features %s in %.1fs", self.Z_sgc.shape, time.time() - t)

    def local_prior(self, seed_idx, seed_y, random_state=0) -> tuple[np.ndarray, LocalModel]:
        m = LocalModel(self.cfg.local_model, random_state=random_state)
        rows = self.local_pos.loc[seed_idx].to_numpy()
        m.fit(self.X_local.iloc[rows], seed_y)
        p = np.full(self.gd.n, np.nan)
        p[self.X_local.index.to_numpy()] = m.predict_proba(self.X_local)
        return p, m

    def run_all(self, seed_idx: np.ndarray, seed_y: np.ndarray, methods=METHODS, random_state=0) -> tuple[dict, dict]:
        """Bot scores (N,) per method + extras (fitted objects / vectors)."""
        scores, extras = {}, {}
        p_local, model = None, None
        if any(m in methods for m in ("local_rf", "trust_score", "trust_score_fixed_w")):
            p_local, model = self.local_prior(seed_idx, seed_y, random_state)
            scores["local_rf"] = p_local
            extras["local_model"] = model
        if "propagation_only" in methods:
            ts = TrustScorer(self, use_local=False).fit(seed_idx, seed_y, random_state)
            scores["propagation_only"] = ts.p_bot
            extras["propagation_only"] = ts
        if "trust_score" in methods:
            ts = TrustScorer(self, use_local=True).fit(seed_idx, seed_y, random_state, p_local_full=p_local)
            scores["trust_score"] = ts.p_bot
            extras["trust_score"] = ts
        if "trust_score_fixed_w" in methods:   # ablation: identical selection + calibration, but hand-set channel weights
            ts = TrustScorer(self, use_local=True, learn_weights=False).fit(seed_idx, seed_y, random_state, p_local_full=p_local)
            scores["trust_score_fixed_w"] = ts.p_bot
            extras["trust_score_fixed_w"] = ts
        if "trustrank" in methods:
            s = trustrank_score(self.A_follow, seed_idx[seed_y == 0], seed_idx[seed_y == 1], alpha=0.85)
            scores["trustrank"] = (1 - s) / 2   # in [0,1] for ranking metrics only; it is not a probability
        if "sgc" in methods:
            lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            lr.fit(self.Z_sgc[seed_idx], seed_y)
            scores["sgc"] = lr.predict_proba(self.Z_sgc)[:, list(lr.classes_).index(1)]
        return scores, extras

    # -- protocols ----------------------------------------------------------------------------------------
    def seed_sweep(self, fractions=None, repeats=None, methods=METHODS) -> list[dict]:
        ec = self.cfg.eval
        fractions = fractions or ec.seed_fractions
        repeats = repeats or ec.repeats
        y_lab = self.gd.y[self.lab]
        rows = []
        for frac in fractions:
            for rep in range(repeats):
                rng = np.random.default_rng(ec.random_state * 1000 + rep)
                pos = sample_seeds(y_lab, frac, rng)
                test_mask = np.ones(len(self.lab), bool)
                test_mask[pos] = False
                t = time.time()
                scores, extras = self.run_all(self.lab[pos], y_lab[pos], methods, random_state=rep)
                for mth, s in scores.items():
                    m = metrics(y_lab[test_mask], s[self.lab][test_mask], thresholded=mth not in NO_THRESHOLD)
                    if mth in extras and isinstance(extras[mth], TrustScorer):
                        m.update({"alpha": extras[mth].alpha, "lam": extras[mth].lam})
                    rows.append({"method": mth, "seed_frac": frac, "repeat": rep, "n_seeds": int(len(pos)), **m})
                log.info("seed sweep frac=%.2f rep=%d done in %.1fs", frac, rep, time.time() - t)
        return rows

    def noise_sweep(self, levels=None, frac=None, repeats=None, methods=METHODS) -> list[dict]:
        ec = self.cfg.eval
        levels = levels or ec.noise_levels
        frac = frac or ec.noise_seed_fraction
        repeats = repeats or ec.repeats
        y_lab = self.gd.y[self.lab]
        rows = []
        for noise in levels:
            for rep in range(repeats):
                rng = np.random.default_rng(ec.random_state * 1000 + 500 + rep)
                pos = sample_seeds(y_lab, frac, rng)
                seed_y = flip_labels(y_lab[pos], noise, rng)
                test_mask = np.ones(len(self.lab), bool)
                test_mask[pos] = False
                scores, _ = self.run_all(self.lab[pos], seed_y, methods, random_state=rep)
                for mth, s in scores.items():
                    m = metrics(y_lab[test_mask], s[self.lab][test_mask], thresholded=mth not in NO_THRESHOLD)
                    rows.append({"method": mth, "noise": noise, "seed_frac": frac, "repeat": rep, **m})
                log.info("noise sweep noise=%.2f rep=%d done", noise, rep)
        return rows


class TrustScorer:
    """The trust score with its hyper-parameters chosen on the seeds only.

    1. channel homophily weights w_c: phi coefficient on seed-seed edges (blended with the hand prior)
    2. (alpha, lam) chosen by stratified inner cross-validation over the seeds (criterion: mean of AUC and AP)
    3. Platt calibration r -> P(bot) fitted on the inner out-of-fold seed residuals (gives a meaningful threshold)
    4. final fit on all seeds: q -> r -> trust = 1/2 - r, p_bot = sigmoid(a r + b)
    """

    def __init__(self, ev: Evaluator, use_local: bool = True, alphas=ALPHA_GRID, lams=LAM_GRID, inner_folds: int = 3,
                 learn_weights: bool = True):
        self.ev, self.use_local, self.alphas, self.inner_folds, self.learn_weights = ev, use_local, alphas, inner_folds, learn_weights
        self.lams = lams if use_local else (0.0,)

    def fit(self, seed_idx: np.ndarray, seed_y: np.ndarray, random_state: int = 0, p_local_full: np.ndarray | None = None) -> "TrustScorer":
        cfg = self.ev.cfg.propagation
        n = self.ev.gd.n
        seed_idx, seed_y = np.asarray(seed_idx), np.asarray(seed_y)
        rng = np.random.default_rng(random_state + 1234)
        grid = [(a, l) for a in self.alphas for l in self.lams]
        oof = {g: np.full(len(seed_idx), np.nan) for g in grid}
        k_folds = min(self.inner_folds, int(min((seed_y == 1).sum(), (seed_y == 0).sum())))
        if k_folds >= 2:
            fold = stratified_folds(seed_y, k_folds, rng)
            for k in range(k_folds):
                tr, va = fold != k, fold == k
                if len(np.unique(seed_y[va])) < 2:
                    continue
                s_tr, y_tr = seed_idx[tr], seed_y[tr]
                p_loc = self.ev.local_prior(s_tr, y_tr, random_state)[0] if self.use_local else None
                w = estimate_channel_weights(self.ev.channels, s_tr, y_tr, cfg.relation_weights)[0] if self.learn_weights else cfg.relation_weights
                P = combine_channels(self.ev.channels, w)
                for a, l in grid:
                    q = build_prior(n, s_tr, y_tr, p_loc, lam=l, seed_residual=cfg.seed_residual)
                    r = propagate(P, q, alpha=a, max_iter=cfg.max_iter, tol=cfg.tol)
                    oof[(a, l)][va] = r[seed_idx[va]]
        # model selection
        self.grid_results = {}
        best, best_crit = None, -np.inf
        for g in grid:
            s = oof[g]
            ok = np.isfinite(s)
            if ok.sum() < 4 or len(np.unique(seed_y[ok])) < 2:
                continue
            auc = roc_auc_score(seed_y[ok], s[ok])
            ap = average_precision_score(seed_y[ok], s[ok])
            crit = 0.5 * (auc + ap)
            self.grid_results[f"a={g[0]},l={g[1]}"] = {"auc": round(float(auc), 4), "ap": round(float(ap), 4)}
            if crit > best_crit + 1e-9:
                best, best_crit = g, crit
        if best is None:  # not enough seeds for inner CV: fall back to config defaults
            best = (cfg.alpha, cfg.lam if self.use_local else 0.0)
        self.alpha, self.lam = float(best[0]), float(best[1])
        # calibration (Platt): residuals are standardised first so the L2 penalty does not flatten the fit
        s = oof.get(best)
        ok = np.isfinite(s) if s is not None else np.zeros(len(seed_idx), bool)
        self.platt = LogisticRegression(C=100.0, max_iter=2000)
        if ok.sum() >= 4 and len(np.unique(seed_y[ok])) > 1:
            self.platt_scale = float(1.0 / max(np.std(s[ok]), 1e-6))
            self.platt.fit((s[ok] * self.platt_scale).reshape(-1, 1), seed_y[ok])
            self.calibrated_on = "inner out-of-fold seed residuals"
        else:
            self.platt_scale = 20.0
            self.platt.fit(np.array([[-10.0], [10.0]]), np.array([0, 1]))
            self.calibrated_on = "fallback (too few seeds)"
        # final fit on all seeds
        if self.use_local and p_local_full is None:
            p_local_full = self.ev.local_prior(seed_idx, seed_y, random_state)[0]
        self.p_local = p_local_full if self.use_local else None
        if self.learn_weights:
            self.weights, self.homophily = estimate_channel_weights(self.ev.channels, seed_idx, seed_y, cfg.relation_weights)
        else:
            self.weights, self.homophily = dict(cfg.relation_weights), {}
        P = combine_channels(self.ev.channels, self.weights)
        self.q = build_prior(n, seed_idx, seed_y, self.p_local, lam=self.lam, seed_residual=cfg.seed_residual)
        self.r, info = propagate(P, self.q, alpha=self.alpha, max_iter=cfg.max_iter, tol=cfg.tol, return_info=True)
        self.iterations, self.converged = info["iterations"], info["converged"]
        self.T = trust_from_residual(self.r)
        self.p_bot = self.calibrate(self.r)
        return self

    @property
    def platt_a(self) -> float:
        """Effective slope on the raw residual: p_bot = sigmoid(platt_a * r + platt_b)."""
        return float(self.platt.coef_[0, 0] * self.platt_scale)

    @property
    def platt_b(self) -> float:
        return float(self.platt.intercept_[0])

    def calibrate(self, r: np.ndarray) -> np.ndarray:
        z = np.clip(self.platt_a * np.asarray(r, dtype=float) + self.platt_b, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def export(self) -> dict:
        return {"alpha": self.alpha, "lam": self.lam, "weights": self.weights, "homophily": self.homophily,
                "platt_a": self.platt_a, "platt_b": self.platt_b,
                "calibrated_on": self.calibrated_on, "grid_results": self.grid_results,
                "iterations": self.iterations, "converged": self.converged}


def summarize(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    df = pd.DataFrame(rows)
    out = []
    for key, sub in df.groupby(list(keys)):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(keys, [k.item() if hasattr(k, "item") else k for k in key]))
        for mcol in ("auc", "ap", "f1", "precision", "recall", "accuracy"):
            col = pd.to_numeric(sub[mcol], errors="coerce")
            rec[mcol] = None if col.isna().all() else float(col.mean())
            rec[mcol + "_std"] = None if col.isna().all() or len(col.dropna()) < 2 else float(col.std(ddof=1))   # sample std over repeats
        rec["n_seeds"] = int(sub["n_seeds"].mean()) if "n_seeds" in sub else None
        for extra in ("alpha", "lam"):
            if extra in sub and sub[extra].notna().any():
                rec[extra + "_mean"] = float(sub[extra].mean())
        out.append(rec)
    return out
