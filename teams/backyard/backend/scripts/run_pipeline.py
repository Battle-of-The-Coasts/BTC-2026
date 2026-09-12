"""Run the full pipeline for one dataset.  Usage: python scripts/run_pipeline.py cresci-2015 [--skip-sweeps]"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import memwatch  # noqa: F401  (aborts if RSS exceeds MEMWATCH_MB)

from trustscore.config import Config
from trustscore.pipeline import run

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["cresci-2015", "mgtab"])
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config(dataset=args.dataset)
    if args.alpha is not None:
        cfg.propagation.alpha = args.alpha
    if args.lam is not None:
        cfg.propagation.lam = args.lam
    out = run(cfg, os.path.join(ROOT, "data", "raw"), os.path.join(ROOT, "data", "processed"), skip_sweeps=args.skip_sweeps)
    print("artifacts:", out, "peak RSS MB:", round(memwatch.peak_mb()))
