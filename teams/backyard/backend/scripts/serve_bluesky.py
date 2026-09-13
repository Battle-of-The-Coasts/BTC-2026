"""Start the Bluesky add-on API. Usage: python scripts/serve_bluesky.py [port]   (default 8010)
Data: data/bluesky/bluesky.sqlite (override with BLUESKY_DB)."""
import logging
import os
import sys

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8010"))
    uvicorn.run("trustscore.bluesky.api:app", host=os.environ.get("HOST", "127.0.0.1"), port=port, workers=1)
