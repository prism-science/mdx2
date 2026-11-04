"""Lightweight step profiling for mdx2 command-line tools: wall time and memory per step.

Disabled by default (every call is a cheap no-op). When enabled, each step records its wall time, the process RSS
after the step and its change, and the process peak RSS so far; code inside a step can attach extra values with
`note(...)` (e.g. lsqr iterations). Records go to a JSON-lines file and/or a summary in the log.
"""

import json
import resource
import sys
import time
from contextlib import contextmanager

from loguru import logger

try:
    import psutil
except ImportError:  # optional: without it only the peak RSS is reported
    psutil = None


def _rss():
    return psutil.Process().memory_info().rss if psutil is not None else None


def _peak_rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024  # bytes on macOS, kilobytes on Linux


def _gb(x):
    return None if x is None else round(x / 1e9, 3)


class Profiler:
    def __init__(self, enabled=False, path=None):
        self.enabled = bool(enabled) or path is not None
        self.path = path
        self.stage = None
        self.records = []
        self._stack = []
        self._t0 = time.perf_counter()
        self._fh = open(path, "w") if path else None

    @contextmanager
    def step(self, name, **info):
        if not self.enabled:
            yield None
            return
        rec = dict(step=name, stage=self.stage, depth=len(self._stack), **info)
        rss0 = _rss()
        t = time.perf_counter()
        self._stack.append(rec)
        try:
            yield rec
        finally:
            self._stack.pop()
            rss1 = _rss()
            rec.update(
                t_start=round(t - self._t0, 4),
                seconds=round(time.perf_counter() - t, 4),
                rss_gb=_gb(rss1),
                rss_delta_gb=_gb(None if rss0 is None else rss1 - rss0),
                peak_rss_gb=_gb(_peak_rss()),
            )
            self.records.append(rec)
            if self._fh is not None:
                self._fh.write(json.dumps(rec) + "\n")
                self._fh.flush()

    def note(self, **kv):
        """Attach values to the innermost open step (repeated keys, e.g. one per batch, become lists)."""
        if not self.enabled or not self._stack:
            return
        rec = self._stack[-1]
        for k, v in kv.items():
            if k in rec:
                rec[k] = (rec[k] if isinstance(rec[k], list) else [rec[k]]) + [v]
            else:
                rec[k] = v

    def wrap(self, obj, method, name=None, **info):
        """Replace obj.method by a version that runs inside a step."""
        if not self.enabled:
            return
        f = getattr(obj, method)

        def wrapped(*a, **k):
            with self.step(name or method, **info):
                return f(*a, **k)

        setattr(obj, method, wrapped)

    def summary(self):
        if not self.enabled:
            return
        total = time.perf_counter() - self._t0
        by = {}
        for r in self.records:
            key = ("  " * r["depth"]) + r["step"]  # nested steps are indented (their time is inside the parent's)
            n, s, m = by.get(key, (0, 0.0, 0.0))
            by[key] = (n + 1, s + r["seconds"], max(m, r["peak_rss_gb"] or 0.0))
        logger.info("Profile: total {:.1f} s, peak RSS {} GB", total, _gb(_peak_rss()))
        for key, (n, s, m) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            logger.info(
                "  {:24s} {:4d} calls  {:9.2f} s  ({:5.1f}%)  peak RSS {:.2f} GB", key, n, s, 100 * s / total, m
            )
        if self._fh is not None:
            self._fh.close()
            self._fh = None


PROFILER = Profiler()


def note(**kv):
    PROFILER.note(**kv)
