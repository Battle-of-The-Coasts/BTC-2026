"""Analytics and graph extraction for the add-on dashboard page (bluesky-addon/dashboard.html).

Everything is computed from the SQLite store on demand; degrees are cached for 30 s because they need a pass
over every edge.
"""
from __future__ import annotations

import time
from collections import defaultdict

from .store import Store

DEG_BINS = [(1, 1, "1"), (2, 3, "2–3"), (4, 10, "4–10"), (11, 30, "11–30"), (31, 100, "31–100"), (101, 300, "101–300"),
            (301, 1000, "301–1k"), (1001, 10 ** 9, ">1k")]
BANDS = ("green", "amber", "red")


def band(t: float | None) -> str:
    if t is None:
        return "grey"
    return "green" if t >= 0.7 else "amber" if t >= 0.4 else "red"


class Analytics:
    def __init__(self, store: Store):
        self.store = store
        self._deg: dict[str, int] | None = None
        self._deg_t = 0.0

    # ---- degrees -------------------------------------------------------------------------------
    def degrees(self) -> dict[str, int]:
        if self._deg is not None and time.time() - self._deg_t < 30:
            return self._deg
        deg: dict[str, int] = defaultdict(int)
        for sql in ("SELECT src, COUNT(*) FROM follows GROUP BY src", "SELECT dst, COUNT(*) FROM follows GROUP BY dst",
                    "SELECT src, COUNT(*) FROM interactions GROUP BY src", "SELECT dst, COUNT(*) FROM interactions GROUP BY dst"):
            for r in self.store.q(sql):
                deg[r[0]] += int(r[1])
        self._deg, self._deg_t = dict(deg), time.time()
        return self._deg

    def crawled(self) -> set[str]:
        return {r[0] for r in self.store.q("SELECT did FROM crawl WHERE followers_at IS NOT NULL")}

    # ---- analytics payload ---------------------------------------------------------------------
    def analytics(self, channel_weights: dict[str, float]) -> dict:
        st = self.store
        hist = [{"bin": b, "lo": b / 20, "hi": (b + 1) / 20, "all": 0, "crawled": 0} for b in range(20)]
        for r in st.q("SELECT CAST(MIN(trust * 20, 19) AS INTEGER) b, COUNT(*) FROM scores GROUP BY b"):
            hist[int(r[0])]["all"] = int(r[1])
        for r in st.q("SELECT CAST(MIN(s.trust * 20, 19) AS INTEGER) b, COUNT(*) FROM scores s JOIN crawl c ON c.did=s.did "
                      "WHERE c.followers_at IS NOT NULL GROUP BY b"):
            hist[int(r[0])]["crawled"] = int(r[1])
        bands = {"all": dict.fromkeys(BANDS + ("grey",), 0), "crawled": dict.fromkeys(BANDS + ("grey",), 0)}
        case = "CASE WHEN trust >= 0.7 THEN 'green' WHEN trust >= 0.4 THEN 'amber' ELSE 'red' END"
        for r in st.q(f"SELECT {case} b, COUNT(*) FROM scores GROUP BY b"):
            bands["all"][r[0]] = int(r[1])
        for r in st.q(f"SELECT {case} b, COUNT(*) FROM scores s JOIN crawl c ON c.did=s.did WHERE c.followers_at IS NOT NULL GROUP BY b"):
            bands["crawled"][r[0]] = int(r[1])
        bands["all"]["grey"] = int(st.one("SELECT COUNT(*) FROM actors a LEFT JOIN scores s ON s.did=a.did WHERE s.did IS NULL")[0])

        deg = self.degrees()
        crawled = self.crawled()
        dh = [{"label": lab, "lo": lo, "hi": hi, "all": 0, "crawled": 0} for lo, hi, lab in DEG_BINS]
        for d, k in deg.items():
            for i, (lo, hi, _) in enumerate(DEG_BINS):
                if lo <= k <= hi:
                    dh[i]["all"] += 1
                    if d in crawled:
                        dh[i]["crawled"] += 1
                    break
        depth = [{"depth": r[0], "n": int(r[1]), "crawled": int(r[2])} for r in st.q(
            "SELECT a.depth, COUNT(*), SUM(CASE WHEN c.followers_at IS NOT NULL THEN 1 ELSE 0 END) FROM actors a "
            "LEFT JOIN crawl c ON c.did=a.did GROUP BY a.depth ORDER BY a.depth")]

        n_follow = int(st.one("SELECT COUNT(*) FROM follows")[0])
        n_mutual_pairs = int(st.one("SELECT COUNT(*) FROM follows f JOIN follows g ON g.src=f.dst AND g.dst=f.src")[0]) // 2
        channels = [{"name": "mutual", "edges": n_mutual_pairs, "weight": channel_weights.get("mutual", 0), "note": "pairs following each other"},
                    {"name": "follow (one-way)", "edges": n_follow - 2 * n_mutual_pairs, "weight": f"{channel_weights.get('follow_in', 0)} in / {channel_weights.get('follow_out', 0)} out", "note": ""}]
        for r in st.q("SELECT kind, COUNT(*), SUM(n) FROM interactions GROUP BY kind ORDER BY kind"):
            channels.append({"name": r[0], "edges": int(r[1]), "weight": f"{channel_weights.get(r[0] + '_in', 0)} in / {channel_weights.get(r[0] + '_out', 0)} out",
                             "note": f"{int(r[2])} events"})

        def top(order: str, n: int = 12):
            rows = st.q(f"""SELECT a.did, a.handle, a.display_name, a.depth, s.trust, sd.label seed
                            FROM scores s JOIN crawl c ON c.did=s.did AND c.followers_at IS NOT NULL JOIN actors a ON a.did=s.did
                            LEFT JOIN seeds sd ON sd.did=s.did ORDER BY s.trust {order} LIMIT ?""", n)
            return [{"did": r["did"], "handle": r["handle"], "name": r["display_name"], "depth": r["depth"], "trust": r["trust"],
                     "seed": r["seed"], "deg": deg.get(r["did"], 0)} for r in rows]

        runs = [dict(r) for r in st.q("SELECT run_id, computed_at, n_nodes, n_edges, n_seeds, iterations, converged FROM runs "
                                      "ORDER BY run_id DESC LIMIT 60")][::-1]
        seeds_by = [dict(r) for r in st.q("SELECT source, label, COUNT(*) n FROM seeds GROUP BY source, label ORDER BY source, label")]
        return {"scores": {"hist": hist, "bands": bands}, "degrees": dh, "depth": depth, "channels": channels,
                "top": {"trusted": top("DESC"), "untrusted": top("ASC")}, "runs": runs, "seeds_by": seeds_by,
                "n_edges_total": n_follow + int(st.one("SELECT COUNT(*) FROM interactions")[0])}

    # ---- accounts table ------------------------------------------------------------------------
    def accounts(self, q: str | None, sort: str, desc: bool, limit: int, offset: int, only_crawled: bool = True) -> dict:
        deg = self.degrees()
        where, args = [], []
        if only_crawled:
            where.append("c.followers_at IS NOT NULL")
        if q:
            where.append("(a.handle LIKE ? OR a.display_name LIKE ?)")
            args += [f"%{q}%", f"%{q}%"]
        w = ("WHERE " + " AND ".join(where)) if where else ""
        col = {"trust": "s.trust", "handle": "a.handle", "depth": "a.depth", "followers": "a.followers_count",
               "posts": "a.posts_count", "created": "a.created_at"}.get(sort, "s.trust")
        rows = self.store.q(f"""SELECT a.did, a.handle, a.display_name, a.depth, a.followers_count, a.follows_count, a.posts_count,
                                       a.created_at, a.labels, s.trust, sd.label seed, c.followers_at
                                FROM actors a LEFT JOIN scores s ON s.did=a.did LEFT JOIN crawl c ON c.did=a.did
                                LEFT JOIN seeds sd ON sd.did=a.did {w}
                                ORDER BY {col} {'DESC' if desc else 'ASC'} NULLS LAST LIMIT ? OFFSET ?""", *args, limit, offset)
        total = int(self.store.one(f"SELECT COUNT(*) FROM actors a LEFT JOIN crawl c ON c.did=a.did {w}", *args)[0])
        out = []
        for r in rows:
            d = dict(r)
            d["deg"] = deg.get(r["did"], 0)
            d["crawled"] = bool(r["followers_at"])
            del d["followers_at"]
            out.append(d)
        return {"rows": out, "total": total}

    # ---- graph extraction ----------------------------------------------------------------------
    def neighbour_ids(self, did: str) -> set[str]:
        out = set()
        for sql in ("SELECT dst FROM follows WHERE src=?", "SELECT src FROM follows WHERE dst=?",
                    "SELECT dst FROM interactions WHERE src=?", "SELECT src FROM interactions WHERE dst=?"):
            out.update(r[0] for r in self.store.q(sql, did))
        out.discard(did)
        return out

    def graph(self, focus: list[str], hops: int = 1, max_nodes: int = 1500, min_degree: int = 2) -> dict:
        deg = self.degrees()
        crawled = self.crawled()
        focus = [f for f in focus if f in deg or self.store.actor(f) is not None]
        if focus:
            selected, seen, frontier = list(focus), set(focus), list(focus)
            for _ in range(max(0, hops)):
                nxt = []
                for d in frontier:
                    nb = sorted(self.neighbour_ids(d), key=lambda x: (x not in crawled, -deg.get(x, 0)))
                    for x in nb:
                        if x in seen:
                            continue
                        if len(selected) >= max_nodes:
                            break
                        seen.add(x); selected.append(x); nxt.append(x)
                frontier = nxt
            mode = "focus"
        else:
            selected = sorted(crawled, key=lambda x: -deg.get(x, 0))[:max_nodes]
            others = sorted((d for d, k in deg.items() if d not in crawled and k >= min_degree), key=lambda d: -deg[d])
            selected += others[:max(0, max_nodes - len(selected))]
            mode = "core"
        sel = set(selected)
        with self.store.lock:
            db = self.store.db
            db.execute("CREATE TEMP TABLE IF NOT EXISTS sel (did TEXT PRIMARY KEY)")
            db.execute("DELETE FROM sel")
            db.executemany("INSERT OR IGNORE INTO sel (did) VALUES (?)", [(d,) for d in selected])
            edges = [{"s": r[0], "t": r[1], "k": "follow", "w": 1} for r in db.execute(
                "SELECT f.src, f.dst FROM follows f JOIN sel a ON a.did=f.src JOIN sel b ON b.did=f.dst")]
            edges += [{"s": r[0], "t": r[1], "k": r[2], "w": int(r[3])} for r in db.execute(
                "SELECT i.src, i.dst, i.kind, i.n FROM interactions i JOIN sel a ON a.did=i.src JOIN sel b ON b.did=i.dst")]
            rows = db.execute("""SELECT a.did, a.handle, a.display_name, a.depth, s.trust, sd.label seed
                                 FROM sel x JOIN actors a ON a.did=x.did LEFT JOIN scores s ON s.did=a.did
                                 LEFT JOIN seeds sd ON sd.did=a.did""").fetchall()
        nodes = [{"id": r["did"], "h": r["handle"], "n": r["display_name"], "d": r["depth"], "t": r["trust"], "s": r["seed"],
                  "c": int(r["did"] in crawled), "deg": deg.get(r["did"], 0), "f": int(r["did"] in set(focus))} for r in rows]
        return {"mode": mode, "hops": hops, "focus": focus, "nodes": nodes, "edges": edges,
                "truncated": len(sel) >= max_nodes, "n_crawled_total": len(crawled), "n_nodes_total": len(deg)}
