# Trust Score — step 1: network-based bot detection on open data

A **trust score** for social-media accounts that judges the *source* (profile metadata, position in the
follow/interaction graph, membership in coordinated clusters) rather than the content it posts.
Step 1 of 3: find bots in open, labelled network datasets, define and validate the score formula, and
expose everything in a web dashboard. Design choices, approximations and hard-coded values are documented in
[`docs/DESIGN.md`](docs/DESIGN.md) (2 pages).

```
backend/    Python package `trustscore` (loaders, features, local model, propagation, clusters, evaluation, FastAPI)
frontend/   React + TypeScript dashboard (sigma.js network explorer, recharts)
data/raw/   downloaded datasets (cresci-2015 from the Bot Repository, MGTAB from GitHub/Google Drive)
data/processed/<dataset>/   pipeline artifacts served by the API
docs/       DESIGN.md (+ PDF)
```

## Quick start

```bash
# 1. backend environment (uv or plain venv)
cd backend && uv venv --python 3.10 .venv && uv pip install --python .venv/bin/python -e .

# 2. datasets (already downloaded in data/raw; see docs for sources)
#    cresci-2015: https://botometer.osome.iu.edu/bot-repository/datasets/cresci-2015/cresci-2015.csv.tar.gz
#    MGTAB:       https://github.com/GraphDetec/MGTAB (Google Drive link in the README)

# 3. run the pipeline (features -> local model -> propagation -> clusters -> evaluation -> artifacts)
.venv/bin/python scripts/run_pipeline.py cresci-2015      # ~5-10 min, add --skip-sweeps for ~1 min
.venv/bin/python scripts/run_pipeline.py mgtab

# 4. API + dashboard
cd ../frontend && npm install && npm run build           # builds into frontend/dist (served by the API)
cd ../backend && .venv/bin/python scripts/serve.py 8000   # http://127.0.0.1:8000
# (development: `npm run dev` in frontend/ proxies /api to the backend on :8000)

# tests
cd backend && .venv/bin/python -m pytest -q
```

## What the score is

For every account `v`, with `q_v` its prior evidence of being a bot (±½ for labelled seeds, `λ·(p_local − ½)`
from a Random-Forest profile/structure model otherwise) and `P` a row-stochastic matrix over typed,
hub-damped edge channels (mutual follow, follower→followee, followee→follower, mentions, replies, …):

```
r = (1 − α)·q + α·P·r          (unique fixed point, |r| ≤ ½)
trust(v) = ½ − r_v ∈ [0, 1]     (1 = trusted, 0 = bot-like)
```

Channel weights are estimated from the label homophily of seed–seed edges (blended with a hand-set prior).
The dashboard decomposes every score into prior + per-channel + per-neighbour terms, lists the local-model
feature contributions, and shows Leiden communities, dense co-following blocks and creation bursts.

## Deploying on the internet

The site is a single FastAPI process serving the built dashboard and the precomputed artifacts (~60 MB, ~1 GB RAM
once loaded), so any 2 GB VM or container host works. `deploy/` contains a multi-stage Dockerfile (builds the
React app, installs the Python package, copies `data/processed` and `docs`), a docker-compose file and a Caddy
config that gives automatic HTTPS and optional password protection.

```bash
# 1. produce the artifacts locally (they are baked into the image)
cd backend && .venv/bin/python scripts/run_pipeline.py cresci-2015 && .venv/bin/python scripts/run_pipeline.py mgtab && cd ..

# 2. on the server (Docker installed, DNS A record of DOMAIN -> server IP, ports 80/443 open)
export DOMAIN=trust.example.com
export BASIC_AUTH_USER=demo BASIC_AUTH_HASH="$(docker run --rm caddy:2 caddy hash-password --plaintext 'choose-a-password')"
docker compose -f deploy/docker-compose.yml up -d --build
```

Alternatives: any PaaS that builds a Dockerfile (Fly.io, Render, Railway; pick a 1–2 GB instance and set the root
Dockerfile path to `deploy/Dockerfile`), or a Hugging Face Space (Docker SDK). Keep the demo password-protected or
pseudonymise `screen_name` before publishing: cresci-2015 and MGTAB are released for research use only.
