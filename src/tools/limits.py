"""Per-key requests-per-minute limit (sliding window, in process)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, rpm: int | None) -> float | None:
        """Record a request; return None if allowed, else the seconds until a slot frees up."""
        if not rpm:
            return None
        with self._lock:
            now = self.clock()
            q = self._hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= rpm:
                return round(60 - (now - q[0]), 2)
            q.append(now)
            return None
