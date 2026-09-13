"""Feature engineering. Two blocks, no tweet text is ever used (the score judges the *source*, not the content):

* profile block   : account metadata (Botometer-style user features) - cresci-2015 raw profiles or MGTAB's 20 property dims
* structural block: position in the follow / interaction graph (degrees, reciprocity, co-following, PageRank, clustering,
                    coreness) computed on the full graph including unlabelled external accounts.
* temporal block  : account-creation bursts (accounts created on the same day), cresci-2015 only.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter

import igraph as ig
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .datasets import GraphData
from .graph import degrees, undirected_union

log = logging.getLogger(__name__)

DEFAULT_COLORS = {
    "profile_link_color": "0084B4", "profile_sidebar_fill_color": "DDEEF6", "profile_background_color": "C0DEED",
    "profile_text_color": "333333", "profile_sidebar_border_color": "C0DEED",
}
# Text-valued fields are never used as text: lang, time_zone and the location/description/name/screen_name/url
# strings mostly encode where/when a sub-collection was gathered (Italian election vs. English fake-follower vendors).
# Only content-free statistics of them are used (presence flags, lengths, digit share, character entropy).
TEXT_FIELDS_NOT_USED_AS_TEXT = ["lang", "time_zone", "location", "description", "name", "screen_name", "url"]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    c = Counter(s)
    n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _flag(series: pd.Series) -> np.ndarray:
    return series.notna().to_numpy().astype(np.float32) if series.dtype == object else (series.fillna(0).astype(float).to_numpy() > 0).astype(np.float32)


def cresci_profile_features(profile: pd.DataFrame, burst_window_days: int = 1) -> pd.DataFrame:
    p = profile
    created = pd.to_datetime(p["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    # 'updated' is the dataset export timestamp (2015-02-14, one value per subset), the closest available proxy for
    # the profile snapshot date; account age is measured against it.
    crawled = pd.to_datetime(p["updated"], errors="coerce", utc=True)
    age_days = ((crawled - created).dt.total_seconds() / 86400.0).clip(lower=1.0).fillna(1.0)
    f = pd.DataFrame(index=p.index)
    for c in ("statuses_count", "followers_count", "friends_count", "favourites_count", "listed_count"):
        v = pd.to_numeric(p[c], errors="coerce").fillna(0).clip(lower=0)
        f[f"log_{c}"] = np.log1p(v)
    followers = pd.to_numeric(p["followers_count"], errors="coerce").fillna(0)
    friends = pd.to_numeric(p["friends_count"], errors="coerce").fillna(0)
    statuses = pd.to_numeric(p["statuses_count"], errors="coerce").fillna(0)
    favs = pd.to_numeric(p["favourites_count"], errors="coerce").fillna(0)
    listed = pd.to_numeric(p["listed_count"], errors="coerce").fillna(0)
    f["log_ff_ratio"] = np.log((followers + 1) / (friends + 1))
    f["log_age_days"] = np.log(age_days)
    f["tweets_per_day"] = statuses / age_days
    f["favs_per_day"] = favs / age_days
    f["followers_per_day"] = followers / age_days
    f["friends_per_day"] = friends / age_days
    f["listed_per_follower"] = listed / (followers + 1)
    for c in ("default_profile", "default_profile_image", "geo_enabled", "profile_use_background_image", "protected",
              "verified", "profile_background_tile"):
        f[c] = _flag(p[c]) if c in p else 0.0
    f["has_url"] = _flag(p["url"])
    f["has_description"] = _flag(p["description"])
    f["has_location"] = _flag(p["location"])
    f["has_time_zone"] = _flag(p["time_zone"])
    f["has_banner"] = _flag(p["profile_banner_url"]) if "profile_banner_url" in p else 0.0
    desc = p["description"].fillna("").astype(str)
    name = p["name"].fillna("").astype(str)
    sn = p["screen_name"].fillna("").astype(str)
    f["description_len"] = desc.str.len()
    f["description_has_url"] = desc.str.contains(r"http|www\.", regex=True).astype(float)
    f["name_len"] = name.str.len()
    f["name_words"] = name.str.split().str.len().fillna(0)
    f["screen_name_len"] = sn.str.len()
    f["screen_name_digits"] = sn.str.count(r"\d")
    f["screen_name_digit_frac"] = f["screen_name_digits"] / sn.str.len().clip(lower=1)
    f["screen_name_trailing_digits"] = sn.map(lambda s: len(re.search(r"\d*$", s).group(0)))
    f["screen_name_upper_frac"] = sn.str.count(r"[A-Z]") / sn.str.len().clip(lower=1)
    f["screen_name_entropy"] = sn.map(_entropy)
    f["name_eq_screen_name"] = (name.str.lower().str.replace(r"\s+", "", regex=True) == sn.str.lower()).astype(float)
    custom = 0
    for col, default in DEFAULT_COLORS.items():
        if col in p:
            custom = custom + (p[col].fillna(default).astype(str).str.upper() != default).astype(int)
    f["customized_colors"] = custom
    # temporal: creation bursts (unsupervised, across all dataset accounts)
    day = created.dt.floor("D")
    counts = day.value_counts()
    f["created_same_day"] = day.map(counts).fillna(1).astype(float) - 1.0
    if burst_window_days > 0:
        valid = day.notna().to_numpy()
        days_sorted = np.sort(day[valid].astype("int64").to_numpy() // (86400 * 10**9))
        d = np.where(valid, day.astype("int64").to_numpy() // (86400 * 10**9), 0)
        lo = np.searchsorted(days_sorted, d - burst_window_days, side="left")
        hi = np.searchsorted(days_sorted, d + burst_window_days, side="right")
        within = (hi - lo - 1).astype(float)
        within[~valid] = 0.0
        f["created_within_window"] = within
    f["created_hour"] = created.dt.hour.fillna(0).astype(float)
    f["created_weekday"] = created.dt.weekday.fillna(0).astype(float)
    return f.astype(np.float32)


def _to_igraph(A_sym: sp.csr_matrix) -> ig.Graph:
    coo = sp.triu(A_sym, k=1).tocoo()
    return ig.Graph(n=A_sym.shape[0], edges=np.column_stack([coo.row, coo.col]), directed=False)


def structural_features(gd: GraphData) -> pd.DataFrame:
    """Network-position features for every node (labelled and external)."""
    N = gd.n
    f = pd.DataFrame(index=np.arange(N))
    if "follow" in gd.relations:
        A = gd.relations["follow"].tocsr()
        Ab = (A > 0).astype(np.float64)
        out_deg, in_deg = degrees(A)
        mutual = Ab.minimum(Ab.T.tocsr())
        mdeg = np.asarray(mutual.sum(axis=1)).ravel()
        f["out_deg"] = out_deg
        f["in_deg"] = in_deg
        f["log_out_deg"] = np.log1p(out_deg)
        f["log_in_deg"] = np.log1p(in_deg)
        f["log_deg_ratio"] = np.log((in_deg + 1) / (out_deg + 1))
        f["mutual_deg"] = mdeg
        f["reciprocity_out"] = np.where(out_deg > 0, mdeg / np.maximum(out_deg, 1), 0.0)
        f["reciprocity_in"] = np.where(in_deg > 0, mdeg / np.maximum(in_deg, 1), 0.0)
        # what kind of accounts do you follow / who follows you
        with np.errstate(divide="ignore", invalid="ignore"):
            f["followee_mean_log_indeg"] = np.where(out_deg > 0, (Ab @ np.log1p(in_deg)) / np.maximum(out_deg, 1), 0.0)
            f["follower_mean_log_outdeg"] = np.where(in_deg > 0, (Ab.T @ np.log1p(out_deg)) / np.maximum(in_deg, 1), 0.0)
            f["cofollow_frac"] = np.where(out_deg > 0, (Ab @ (in_deg >= 2).astype(float)) / np.maximum(out_deg, 1), 0.0)
            f["cofollowed_frac"] = np.where(in_deg > 0, (Ab.T @ (out_deg >= 2).astype(float)) / np.maximum(in_deg, 1), 0.0)
        # PageRank on the directed follow graph (popularity conferred by followers)
        coo = Ab.tocoo()
        g = ig.Graph(n=N, edges=np.column_stack([coo.row, coo.col]), directed=True)
        pr = np.asarray(g.pagerank(directed=True, damping=0.85))
        f["log_pagerank"] = np.log(pr * N + 1e-12)
        del g
    for name, A in gd.relations.items():
        if name == "follow":
            continue
        o, i = degrees(A)
        if name in gd.symmetric_relations:
            f[f"log_{name}_deg"] = np.log1p(o)
        else:
            f[f"log_{name}_out"] = np.log1p(o)
            f[f"log_{name}_in"] = np.log1p(i)
    U = undirected_union(gd)
    g = _to_igraph(U)
    f["clustering"] = np.asarray(g.transitivity_local_undirected(mode="zero"))
    f["coreness"] = np.asarray(g.coreness(mode="all"), dtype=float)
    f["log_union_deg"] = np.log1p(np.asarray(g.degree(), dtype=float))
    del g
    return f.astype(np.float32)


def assemble_features(gd: GraphData, burst_window_days: int = 1, use_content: bool = False):
    """Return (X_local: DataFrame for nodes that have a profile block, X_all: DataFrame for every node, blocks).

    X_local feeds the local (Random Forest) model; X_all feeds graph baselines (SGC) and includes a missing-profile flag.
    """
    struct = structural_features(gd)
    blocks = {"structural": list(struct.columns)}
    if gd.profile is not None:
        prof = cresci_profile_features(gd.profile, burst_window_days=burst_window_days)
        blocks["profile"] = list(prof.columns)
        blocks["temporal"] = [c for c in prof.columns if c.startswith("created_")]
        blocks["profile"] = [c for c in blocks["profile"] if c not in blocks["temporal"]]
    elif gd.X_pre is not None:
        prof = pd.DataFrame(gd.X_pre, index=np.arange(gd.n), columns=gd.X_pre_names)
        blocks["profile"] = list(prof.columns)
        if use_content and gd.X_content is not None:
            cont = pd.DataFrame(gd.X_content, index=np.arange(gd.n), columns=[f"content_{i}" for i in range(gd.X_content.shape[1])])
            prof = pd.concat([prof, cont], axis=1)
            blocks["content"] = list(cont.columns)
    else:
        prof = pd.DataFrame(index=np.arange(0))
    has_profile = np.zeros(gd.n, dtype=bool)
    has_profile[prof.index.to_numpy()] = True
    X_local = pd.concat([prof, struct.loc[prof.index]], axis=1)
    prof_all = prof.reindex(np.arange(gd.n))
    X_all = pd.concat([prof_all.fillna(0.0), struct], axis=1)
    X_all["profile_missing"] = (~has_profile).astype(np.float32)
    log.info("features: local %s, all %s", X_local.shape, X_all.shape)
    return X_local, X_all, blocks
