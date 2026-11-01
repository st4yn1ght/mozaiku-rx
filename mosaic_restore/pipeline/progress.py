"""Indikator kemajuan sederhana di terminal."""

from __future__ import annotations

import sys
import time


def fmt_time(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):
        return "--:--:--"
    seconds = int(max(0, seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class Progress:
    def __init__(self, label: str, total: int | None = None, stream=sys.stderr, interval: float = 0.5,
                 enabled: bool = True):
        self.label = label
        self.total = total
        self.stream = stream
        self.interval = interval
        self.enabled = enabled
        self.done = 0
        self.t0 = time.monotonic()
        self._last = 0.0

    def update(self, n: int = 1, extra: str = "") -> None:
        self.done += n
        now = time.monotonic()
        if self.enabled and (now - self._last >= self.interval):
            self._last = now
            self._render(now, extra)

    def _render(self, now: float, extra: str = "") -> None:
        elapsed = max(1e-6, now - self.t0)
        fps = self.done / elapsed
        if self.total:
            pct = 100.0 * min(self.done, self.total) / self.total
            eta = (self.total - self.done) / fps if fps > 0 else float("inf")
            msg = f"\r{self.label}: {self.done}/{self.total} ({pct:5.1f}%) | {fps:6.1f} fps | sisa {fmt_time(eta)}"
        else:
            msg = f"\r{self.label}: {self.done} frame | {fps:6.1f} fps | {fmt_time(elapsed)}"
        if extra:
            msg += f" | {extra}"
        self.stream.write(msg + "   ")
        self.stream.flush()

    def close(self) -> float:
        now = time.monotonic()
        if self.enabled:
            self._render(now)
            self.stream.write("\n")
            self.stream.flush()
        return now - self.t0
