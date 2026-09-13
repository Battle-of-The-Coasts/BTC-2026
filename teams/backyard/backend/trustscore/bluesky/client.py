"""Minimal client for the public Bluesky AppView (https://public.api.bsky.app), no authentication.

All calls go through one token-bucket rate limiter (public AppView limit is ~3000 requests / 5 min per IP;
we stay well below) and retry on 429/5xx with exponential back-off.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import requests

PUBLIC_API = "https://public.api.bsky.app/xrpc"


class RateLimiter:
    def __init__(self, per_second: float = 4.0, burst: int = 8):
        self.rate = float(per_second)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.rate
                time.sleep(wait)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


class BskyError(RuntimeError):
    def __init__(self, status: int, msg: str):
        super().__init__(f"{status}: {msg}")
        self.status = status


class BskyClient:
    def __init__(self, base: str = PUBLIC_API, per_second: float = 4.0, timeout: float = 20.0,
                 user_agent: str = "trustscore-bluesky-addon/0.1 (research; contact via repo)"):
        self.base = base.rstrip("/")
        self.limiter = RateLimiter(per_second)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = user_agent
        self.calls = 0

    def get(self, method: str, **params) -> dict[str, Any]:
        params = {k: v for k, v in params.items() if v is not None}
        backoff = 1.0
        for attempt in range(5):
            self.limiter.acquire()
            self.calls += 1
            try:
                r = self.s.get(f"{self.base}/{method}", params=params, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == 4:
                    raise BskyError(0, str(e))
                time.sleep(backoff); backoff *= 2
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                ra = r.headers.get("ratelimit-reset")
                wait = backoff
                if r.status_code == 429 and ra:
                    try:
                        wait = max(1.0, float(ra) - time.time())
                    except ValueError:
                        pass
                time.sleep(min(wait, 60)); backoff *= 2
                continue
            try:
                msg = r.json().get("message", r.text)
            except ValueError:
                msg = r.text
            raise BskyError(r.status_code, msg)
        raise BskyError(0, "unreachable")

    # ---- typed helpers -------------------------------------------------------------------------
    def get_profiles(self, actors: list[str]) -> list[dict]:
        out = []
        for i in range(0, len(actors), 25):
            out.extend(self.get("app.bsky.actor.getProfiles", actors=actors[i:i + 25]).get("profiles", []))
        return out

    def get_profile(self, actor: str) -> dict:
        return self.get("app.bsky.actor.getProfile", actor=actor)

    def _paged(self, method: str, key: str, actor: str, max_pages: int, page_size: int = 100) -> tuple[list[dict], bool]:
        items, cursor, truncated = [], None, False
        for page in range(max_pages):
            d = self.get(method, actor=actor, limit=page_size, cursor=cursor)
            items.extend(d.get(key, []))
            cursor = d.get("cursor")
            if not cursor:
                break
        else:
            truncated = cursor is not None
        return items, truncated

    def followers(self, actor: str, max_pages: int = 3) -> tuple[list[dict], bool]:
        return self._paged("app.bsky.graph.getFollowers", "followers", actor, max_pages)

    def follows(self, actor: str, max_pages: int = 3) -> tuple[list[dict], bool]:
        return self._paged("app.bsky.graph.getFollows", "follows", actor, max_pages)

    def author_feed(self, actor: str, limit: int = 100) -> list[dict]:
        d = self.get("app.bsky.feed.getAuthorFeed", actor=actor, limit=min(limit, 100), filter="posts_with_replies")
        return d.get("feed", [])


def did_of_uri(uri: str | None) -> str | None:
    """at://did:plc:xxx/app.bsky.feed.post/yyy -> did:plc:xxx"""
    if not uri or not uri.startswith("at://"):
        return None
    return uri[5:].split("/", 1)[0] or None


def extract_interactions(actor_did: str, feed: list[dict]) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """From an author feed return (src_did, dst_did, kind) triples plus any actor profiles seen on the way.
    kinds: reply (actor replied to dst), repost (actor reposted dst's post), quote (actor quoted dst), mention."""
    edges: list[tuple[str, str, str]] = []
    seen: dict[str, dict] = {}

    def note(a: dict | None):
        if a and a.get("did"):
            seen.setdefault(a["did"], a)

    for item in feed:
        post = item.get("post") or {}
        author = post.get("author") or {}
        note(author)
        reason = item.get("reason") or {}
        if reason.get("$type", "").endswith("#reasonRepost"):
            if author.get("did") and author["did"] != actor_did:
                edges.append((actor_did, author["did"], "repost"))
            continue   # the post itself is someone else's; do not attribute its replies/mentions to the actor
        if author.get("did") != actor_did:
            continue
        rec = post.get("record") or {}
        parent = (rec.get("reply") or {}).get("parent") or {}
        pdid = did_of_uri(parent.get("uri"))
        if pdid and pdid != actor_did:
            edges.append((actor_did, pdid, "reply"))
            note(((item.get("reply") or {}).get("parent") or {}).get("author"))
        emb = post.get("embed") or {}
        recs = []
        if emb.get("$type", "").endswith("embed.record#view"):
            recs.append(emb.get("record") or {})
        elif emb.get("$type", "").endswith("embed.recordWithMedia#view"):
            recs.append(((emb.get("record") or {}).get("record")) or {})
        for r in recs:
            qa = r.get("author") or {}
            qdid = qa.get("did") or did_of_uri(r.get("uri"))
            if qdid and qdid != actor_did:
                edges.append((actor_did, qdid, "quote"))
                note(qa)
        for facet in rec.get("facets") or []:
            for f in facet.get("features") or []:
                if f.get("$type", "").endswith("facet#mention") and f.get("did") and f["did"] != actor_did:
                    edges.append((actor_did, f["did"], "mention"))
    return edges, seen
