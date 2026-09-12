"""Start the API (and the built frontend if frontend/dist exists).  Usage: python scripts/serve.py [port]"""
import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")   # set HOST=0.0.0.0 inside a container / behind a reverse proxy
    uvicorn.run("trustscore.api:app", host=host, port=port, reload=False, workers=1)
