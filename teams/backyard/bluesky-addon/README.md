# Trust Score for Bluesky — browser add-on (step 3, live)

Unlike the Reddit/X mock-ups, this one is real: it reads the accounts on any `bsky.app` page, asks a local
server to crawl their neighbourhood breadth-first from the **public** Bluesky API (no login, no key), runs
the step-1 propagation on the stored graph and puts the score badge next to every account name.

```
bluesky-addon/            Chrome extension (Manifest V3): content.js, background.js, popup.html/js, overlay.css
backend/trustscore/bluesky/  client.py (rate-limited AppView calls), store.py (SQLite), crawler.py (BFS),
                             scoring.py (propagation + explanations), api.py (FastAPI on :8010)
backend/scripts/serve_bluesky.py
data/bluesky/bluesky.sqlite  the persisted graph (gitignored)
```

## Run

```bash
cd backend && PYTHONPATH=$PWD .venv/bin/python scripts/serve_bluesky.py 8010
```

Then in Chrome: `chrome://extensions` → Developer mode → *Load unpacked* → pick `bluesky-addon/`.
Open https://bsky.app (logged in or not). Badges appear as accounts get crawled; the toolbar popup shows
the crawl status, the seeds and the crawl/scoring parameters.

## What happens

1. **Page scan.** `content.js` finds every `/profile/<handle>` link (feed, thread, notifications, follower
   lists …), one badge per post/list item and account. Accounts within ~1500 px of the viewport are sent with
   priority 0, the rest ("upcoming users") with priority 1. A MutationObserver + scroll listener repeat this as
   the virtualised feed loads.
2. **Lookup.** `POST /bsky/lookup` resolves handles to DIDs (one batched `getProfiles` call for unknown ones),
   returns the stored score for accounts already in the graph, and enqueues the others.
3. **Breadth-first crawl** (`crawler.py`). The frontier is a table ordered by `(priority, depth, arrival)`, so
   every account requested by the page (depth 0) is fully expanded before any neighbour (depth 1) is touched,
   and the queue survives restarts. Per account: profile, followers (≤ `list_pages`×100), follows (same),
   last `feed_limit` posts → `reply`, `repost`, `quote`, `mention` edges. Then up to `expand_per_node`
   neighbours (mutual follows first, then interaction partners, then one-way follows) are enqueued at depth+1
   with background priority, while `depth < max_depth`. A crawled account is not fetched again for `ttl_hours`.
4. **Storage** (`store.py`, SQLite WAL): `actors`, `follows(src,dst)`, `interactions(src,dst,kind,n)`,
   `crawl` (timestamps, truncation flags, errors), `queue`, `seeds`, `scores`, `runs`, `settings`.
   Re-crawling an account replaces the interactions it emitted (the feed is a snapshot) and adds follows.
5. **Scoring** (`scoring.py`). The store is loaded as sparse matrices and the unchanged step-1 formula runs:
   `r = (1−α) q + α P r`, `trust = ½ − r`, with hub-damped channels `mutual / follow_in / follow_out /
   reply_* / repost_* / quote_* / mention_*` and hand-set homophily weights (see `ScoringSettings`).
   The prior `q` is `+½` / `−½` for seeds and `λ (p_local − ½)` otherwise, with `p_local` the transparent profile
   prior of `local_prior.py` (account age, handle pattern, avatar, follower/follow ratio, posting rate, moderation
   labels; every weight is listed in that file and in the dashboard's Method tab). It is re-run at most every 20 s
   while the crawl progresses, and immediately when a seed changes.
6. **Seeds.** *List*: "Import news outlets + verified accounts" resolves the ~230 outlet handles of
   `trusted_lists.py` (unknown ones skipped) and every verified account returned by 25 news-related actor
   searches, marks them trusted and queues their crawl. *Rules* (re-derived before every run, thresholds in the
   popup / dashboard): verified account ⇒ trusted; created < 7 days ago or < 20 followers ⇒ untrusted; moderation
   label (spam, bot, impersonation, …) ⇒ untrusted. *Manual*: "Mark trusted / untrusted" in any popover or the
   dashboard; manual labels always win. *Random* draw is still available as an option.
7. **Popover.** Same layout as the Reddit mock-up: profile facts, "Why this score" rows (exact linear
   decomposition: prior + one row per channel, summing to the score), most influential neighbours, flags
   (seed, moderation label, truncated lists, crawl state), seed buttons.

## API (all under `/bsky`)

| endpoint | purpose |
|---|---|
| `POST /lookup {idents, priority}` | scores + crawl state; enqueues unknown/stale accounts |
| `GET /explain/{handle}` | popover payload |
| `GET /ego/{handle}?limit=60` | ego graph (nodes with scores, typed edges) |
| `GET/POST /seeds` | `{action: random\|set\|remove\|clear, ident, label, k}` |
| `GET/POST /settings` | `{crawler: {...}, scoring: {...}}` |
| `POST /recompute`, `GET /status`, `GET /health` | |

## Approximations (to be listed in DESIGN.md)

* Lists are capped (`list_pages`×100 followers/follows per account) and only the last 100 posts are read; the
  flag "Lists truncated" is shown when a cap was hit. Likes are not crawled (one call per post).
* The profile prior is hand-set evidence (no Bluesky labels to train on); leaves seen only in a list carry handle,
  display name, avatar and creation date, not counts. Uncrawled accounts with fewer than 3 stored edges are shown
  as "not enough graph" (grey `?`) rather than scored.
* Channel weights are the step-1 hand-set prior, not re-estimated from Bluesky seeds.
* Random trusted seeds (50 by default) are a placeholder chosen by the user; scores are relative to that choice.
* BFS defaults: max depth 2, 25 expansions per page account, 6 per deeper node (≈ 175 accounts, ~1400 API calls,
  per page account); the queue is ordered by depth, so every depth-1 layer finishes before depth 2 starts.
* Public AppView rate limit (~3000 req / 5 min per IP) → 4 req/s; one account costs ~8 calls, so the default
  settings crawl ~30 accounts a minute.

## Dashboard

`http://127.0.0.1:8010/addon/dashboard.html` (buttons "Open dashboard" / "Graph of this page" in the popup, "Open in
graph" in every popover): Overview (score and degree distributions, bands, graph growth per run, depth, channels,
extremes), Network (canvas force layout of an ego graph or the crawled core, colour by trust / depth / seed /
crawled, contrast slider, click → explanation drawer with seed buttons), Accounts (sortable table), Seeds
(draw / add / flip / remove, propagation parameters and channel weights), Method. Endpoints: `/bsky/analytics`,
`/bsky/graph?focus=&hops=&max_nodes=`, `/bsky/accounts`.

## Tests

```bash
cd backend && PYTHONPATH=$PWD .venv/bin/python -m pytest -q tests/test_bluesky.py
```
(fake AppView, no network: interaction extraction, BFS order and priorities, persistence, propagation and
exact explanation sums, API round trip).
