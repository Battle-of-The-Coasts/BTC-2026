"""Import-time memory watchdog: aborts the process if resident memory exceeds MEMWATCH_MB (default 2500)."""
import os, threading, time, sys


def _rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0.0


_peak = 0.0


def _watch(limit):
    global _peak
    while True:
        r = _rss_mb()
        _peak = max(_peak, r)
        if r > limit:
            sys.stderr.write(f"\n[memwatch] RSS {r:.0f} MB > limit {limit} MB - aborting to protect the machine\n")
            sys.stderr.flush()
            os._exit(137)
        time.sleep(0.2)


def peak_mb():
    return _peak


limit = float(os.environ.get("MEMWATCH_MB", "2500"))
threading.Thread(target=_watch, args=(limit,), daemon=True).start()
