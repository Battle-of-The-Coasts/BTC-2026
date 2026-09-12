# Trust Score — browser add-on mock-ups

Static, realistic fakes of what the step-3 browser add-on will inject into a social platform page.
No build step: open `index.html` (or `reddit.html` / `twitter.html`) directly in a browser, or serve the
folder (`python3 -m http.server 8765 --directory mockups`).

| file | what |
|---|---|
| `index.html` | landing page with links, legend and a description of what each page demonstrates |
| `reddit.html` | r/personalfinance thread (light Reddit 2025 UI) about an "11.5 % APY savings app"; a burst of same-day accounts pushes the product |
| `twitter.html` | X "For you" timeline (dark theme): a journalist's thread on the same product, bought-follower replies, an overnight "breaking news" account |
| `overlay.css` / `overlay.js` | the add-on pieces shared by both pages: badge, avatar ring, explanation popover, top banner with guide + on/off switch, side summary card, "how does it work" modal |

## What the overlay shows

* **Badge** next to each username: trust score 0–100 (100 × trust, trust = ½ − r from the propagation), green ≥ 70, amber 40–69, red < 40, grey = declared bot / not scored.
* **Popover** (hover or click): profile facts; "Why this score" with one signed bar per term (own evidence, one row per network channel, cluster) that sum exactly to the score; the most influential neighbours with their effect; flags (creation burst, dense block, seed); footer with the computation date and the "no post text is used" reminder.
* **Summary card** in the side column: band counts for the page and the coordinated block active on it.
* **Banner switch**: hides every injected element to compare with the bare page. **Guide**: four numbered examples per page with "Show me" buttons.

All numbers are illustrative but follow the real score's structure (see `docs/DESIGN.md`): channel names are the Reddit channels planned for step 2 (reply, co-comment, co-subreddit) and the Twitter channels used in step 1 (mutual, followers, following, mentions/replies).

Each page hands its own account table to `TrustOverlay.init({ platformLabel, computedOn, dark, how, accounts })`; an account is
`{ handle, sub, score, pbot, facts, why: [[label, contribution, note]…], neighbours: [[handle, relation, score, effect]…], flags: [[text, class]…], note }`
or `{ handle, sub, exempt: '…', exemptShort: 'bot' }` for accounts that are not scored.
