"""Transparent profile prior for every node: the step-1 term q = lam * (p_local - 1/2) without a trained model.

There are no Bluesky labels to train on, so p_local is a hand-set evidence sum, each item documented here so the
design doc can list it. Positive evidence = bot-like. p_local = 1/2 + 1/2 * tanh(sum) (0.5 when nothing is known).
Fields come from the AppView profile (crawled accounts) or from the follower/follow listing (leaves: handle,
display name, avatar, creation date, labels; no counts).
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone

BOT_LABELS = {"spam", "impersonation", "scam", "!hide", "!warn", "bot", "engagement-farming", "inauthentic"}

# (name shown in the explanation, evidence) -- see DESIGN approximations table
EVIDENCE = {
    "created < 30 days ago": 0.6, "created < 180 days ago": 0.2, "created > 2 years ago": -0.3,
    "default handle ending in digits": 0.3, "custom domain handle": -0.3,
    "no avatar": 0.3, "no display name": 0.15,
    "follows 5x more than followers": 0.4, "large audience": -0.2, "no posts": 0.2,
    "> 50 posts/day": 0.4, "> 20 posts/day": 0.2, "moderation label": 1.0,
}


def _age_days(created_at: str | None, now: datetime | None = None) -> float | None:
    if not created_at:
        return None
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0.0, ((now or datetime.now(timezone.utc)) - dt).total_seconds() / 86400)


def profile_evidence(a, now: datetime | None = None) -> tuple[float, list[tuple[str, float]]]:
    """a: actors row (sqlite3.Row or dict). Returns (p_local, [(evidence name, value), ...])."""
    g = (lambda k: a[k]) if not isinstance(a, dict) else a.get
    comps: list[tuple[str, float]] = []

    def add(name):
        comps.append((name, EVIDENCE[name]))

    age = _age_days(g("created_at"), now)
    if age is not None:
        if age < 30:
            add("created < 30 days ago")
        elif age < 180:
            add("created < 180 days ago")
        elif age > 730:
            add("created > 2 years ago")
    handle = g("handle") or ""
    known_profile = bool(handle)      # nodes seen only as a DID in an interaction carry no profile information
    if handle.endswith(".bsky.social"):
        if re.search(r"\d{3,}$", handle[: -len(".bsky.social")]):
            add("default handle ending in digits")
    elif "." in handle and handle != "handle.invalid":
        add("custom domain handle")
    if known_profile:
        if not g("avatar"):
            add("no avatar")
        if not (g("display_name") or "").strip():
            add("no display name")
    followers, follows, posts = g("followers_count"), g("follows_count"), g("posts_count")
    if followers is not None and follows is not None:
        if follows > 200 and follows > 5 * followers:
            add("follows 5x more than followers")
        if followers > 1000 and followers > 10 * max(follows, 1):
            add("large audience")
    if posts is not None:
        if posts == 0:
            add("no posts")
        elif age:
            rate = posts / max(age, 1.0)
            if rate > 50:
                add("> 50 posts/day")
            elif rate > 20:
                add("> 20 posts/day")
    labels = g("labels")
    if labels:
        try:
            if set(json.loads(labels)) & BOT_LABELS:
                add("moderation label")
        except (ValueError, TypeError):
            pass
    e = sum(v for _, v in comps)
    return 0.5 + 0.5 * math.tanh(e), comps
