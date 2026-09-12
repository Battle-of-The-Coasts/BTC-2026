"""Render docs/DESIGN.template.md -> docs/DESIGN.md (placeholders filled from data/processed/*) -> DESIGN.html -> DESIGN.pdf.
Usage: python scripts/build_docs.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import markdown

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = os.path.join(ROOT, "docs")
PROCESSED = os.path.join(ROOT, "data", "processed")
TEMPLATE = os.path.join(DOCS, "DESIGN.template.md")
MD = os.path.join(DOCS, "DESIGN.md")
HTML = os.path.join(DOCS, "DESIGN.html")
PDF = os.path.join(DOCS, "DESIGN.pdf")

CSS = """
@page { size: A4; margin: 9mm 10mm; }
body { font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; font-size: 8.4pt; line-height: 1.24; color: #111; }
h1 { font-size: 13pt; margin: 0 0 2pt 0; } h2 { font-size: 10pt; margin: 5pt 0 2pt 0; border-bottom: 1px solid #999; }
p { margin: 0 0 3.5pt 0; } ul, ol { margin: 0 0 3.5pt 0; padding-left: 15pt; } li { margin: 0 0 1pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.4pt; margin: 2pt 0 3pt 0; }
th, td { border: 1px solid #bbb; padding: 1pt 3.5pt; vertical-align: top; text-align: left; }
th { background: #eee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 7.9pt; background: #f3f3f3; padding: 0 2px; }
em { color: #222; }
"""


def f(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def collect(ds: str, short: str) -> dict:
    d = os.path.join(PROCESSED, ds)
    m = json.load(open(os.path.join(d, "metrics.json")))
    c = json.load(open(os.path.join(d, "clusters.json")))
    v = {}
    codes = {"trust_score": "ts", "local_rf": "rf", "propagation_only": "po", "sgc": "sgc", "trustrank": "tr", "trust_score_fixed_w": "fw"}
    for r in m["seed_sweep"]:
        code = codes.get(r["method"])
        if code and abs(r["seed_frac"] - 0.01) < 1e-9:
            v[f"{short}_{code}_1"] = f(r["auc"])
        if code and abs(r["seed_frac"] - 0.10) < 1e-9:
            v[f"{short}_{code}_10"] = f(r["auc"])
            v[f"{short}_{code}_f1_10"] = f(r.get("f1"))
    for r in m["noise_sweep"]:
        code = codes.get(r["method"])
        if code and abs(r["noise"] - 0.30) < 1e-9:
            v[f"{short}_{code}_n30"] = f(r["auc"])
    summ = json.load(open(os.path.join(d, "summary.json"))) if os.path.exists(os.path.join(d, "summary.json")) else {}
    sfp = (summ.get("dataset") or {}).get("shared_followee_purity") or {}
    v[f"{short}_shared_purity"] = f"{100 * sfp['mean_purity']:.1f} %" if sfp else "n/a"
    v[f"{short}_shared_n"] = f"{sfp['n_shared_followees']:,}" if sfp else "n/a"
    cf = m["crossfit"]["trust_score"]
    v[f"{short}_cf_auc"], v[f"{short}_cf_ap"], v[f"{short}_cf_f1"] = f(cf["auc"]), f(cf["ap"]), f(cf["f1"])
    sel = m.get("selected_params", {}).get("0", {})
    v[f"{short}_alpha"], v[f"{short}_lam"] = f(sel.get("alpha"), 1), f(sel.get("lam"), 1)
    w = m["channel_weights"].get("0", {})
    for k in ("mutual", "follow_in", "follow_out", "reply_in", "hashtag", "url"):
        v[f"{short}_w_{k}"] = f(w.get(k), 2)
    v[f"{short}_w_reply"] = f(w.get("reply_in"), 2)
    # single-factor calibration check from the stored per-fold residuals (no pipeline rerun needed)
    try:
        import numpy as np
        z = np.load(os.path.join(d, "propagation.npz"))
        lab, y, fold_of = z["lab"], z["y"], z["fold_of"]
        r_all = np.zeros(len(y)); p_all = np.zeros(len(y)); p0 = []
        for k in range(int(fold_of.max()) + 1):
            r = z[f"fold{k}_r"].astype(float); a, b = z[f"fold{k}_params"][2], z[f"fold{k}_params"][3]
            test = lab[fold_of == k]
            r_all[test] = r[test]; p_all[test] = 1 / (1 + np.exp(-np.clip(a * r[test] + b, -60, 60)))
            p0.append(1 / (1 + np.exp(-b)))
        yy = y[lab]
        def f1(pred):
            tp = ((pred == 1) & (yy == 1)).sum(); fp = ((pred == 1) & (yy == 0)).sum(); fn = ((pred == 0) & (yy == 1)).sum()
            return 2 * tp / max(1, 2 * tp + fp + fn)
        v[f"{short}_f1_raw"] = f(f1((r_all[lab] > 0).astype(int)))
        v[f"{short}_f1_cal"] = f(f1((p_all[lab] > 0.5).astype(int)))
        v[f"{short}_p_at_zero"] = f"{min(p0):.2f}–{max(p0):.2f}"
    except Exception as e:  # noqa: BLE001
        v[f"{short}_f1_raw"] = v[f"{short}_f1_cal"] = v[f"{short}_p_at_zero"] = "n/a"
        print("calibration check failed:", e)
    comms = c["communities"]
    sizes = [x["size"] for x in comms]
    pur = [max(x["true_bot_share"], 1 - x["true_bot_share"]) for x in comms]
    v[f"{short}_n_comm"] = str(len(comms))
    v[f"{short}_purity"] = f(sum(s * p for s, p in zip(sizes, pur)) / max(1, sum(sizes)))
    if c["dense_blocks"]:
        b = max(c["dense_blocks"], key=lambda b: b["density"])
        v[f"{short}_block0_size"] = str(b["size"])
        v[f"{short}_block0_bot"] = f"{100 * b['true_bot_share']:.0f} %"
    if c["burst_days"]:
        v[f"{short}_burst_top_day"] = c["burst_days"][0]["day"]
        v[f"{short}_burst_top_n"] = str(c["burst_days"][0]["n_accounts"])
    return v


if __name__ == "__main__":
    values = {}
    for ds, short in (("cresci-2015", "cresci"), ("mgtab", "mgtab")):
        if os.path.exists(os.path.join(PROCESSED, ds, "metrics.json")):
            values.update(collect(ds, short))
    text = open(TEMPLATE).read()
    missing = sorted(set(re.findall(r"{{(\w+)}}", text)) - set(values))
    for k in missing:
        values[k] = "n/a"
    text = re.sub(r"{{(\w+)}}", lambda mm: values[mm.group(1)], text)
    with open(MD, "w") as fh:
        fh.write(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    with open(HTML, "w") as fh:
        fh.write(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>")
    print("wrote", MD, "(missing placeholders:", missing, ")")
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or "/opt/google/chrome/chrome"
    if not os.path.exists(chrome):
        print("Chrome not found; HTML only")
        sys.exit(0)
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", f"file://{HTML}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    try:
        from pypdf import PdfReader
        print("wrote", PDF, "pages:", len(PdfReader(PDF).pages))
    except Exception:
        print("wrote", PDF)
