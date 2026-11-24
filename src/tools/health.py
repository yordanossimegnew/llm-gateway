"""Per-deployment health: circuit breaker, rate-limit cooldowns and a latency average.

closed --(N transient failures in a row)--> open --(cooldown)--> half_open --(1 probe)--> closed
                                              ^                                   |
                                              +------ probe fails (cooldown x2) --+

A 429 doesn't trip the circuit: the deployment is healthy, just busy. It gets a cooldown for its
Retry-After instead, and the planner skips it until then.
Auth and not-found errors open the circuit at once: retrying a bad key won't help.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class DeploymentHealth:
    state: str = "closed"                # closed | open | half_open
    failures: int = 0                    # consecutive circuit-tripping failures
    opened_at: float | None = None
    cooldown_s: float = 0.0
    probe_in_flight: bool = False
    rate_limited_until: float = 0.0
    latency_ms: float | None = None      # exponentially weighted moving average of successful calls
    last_error: str | None = None
    last_error_at: float | None = None
    successes: int = 0
    errors: int = 0
    trips: int = 0
    history: list = field(default_factory=list)  # recent (time, ok) for the UI sparkline


class HealthBoard:
    def __init__(self, failures: int = 3, cooldown_s: float = 30, max_cooldown_s: float = 300, clock=time.time):
        self.threshold = failures
        self.base_cooldown = cooldown_s
        self.max_cooldown = max_cooldown_s
        self.clock = clock
        self._h: dict[str, DeploymentHealth] = {}
        self._lock = threading.Lock()

    def _get(self, name: str) -> DeploymentHealth:
        return self._h.setdefault(name, DeploymentHealth())

    def _refresh(self, h: DeploymentHealth) -> None:
        if h.state == "open" and h.opened_at is not None and self.clock() - h.opened_at >= h.cooldown_s:
            h.state, h.probe_in_flight = "half_open", False

    def available(self, name: str) -> tuple[bool, str | None]:
        """Whether the planner should try this deployment now, and if not, why."""
        with self._lock:
            h = self._get(name)
            self._refresh(h)
            now = self.clock()
            if h.state == "open":
                return False, f"circuit open ({h.cooldown_s - (now - h.opened_at):.0f}s left)"
            if h.state == "half_open" and h.probe_in_flight:
                return False, "circuit half-open, probe in flight"
            if h.rate_limited_until > now:
                return False, f"rate limited ({h.rate_limited_until - now:.1f}s left)"
            return True, None

    def wait_time(self, name: str) -> float:
        """Seconds until a rate-limit cooldown ends (0 if none)."""
        with self._lock:
            return max(self._get(name).rate_limited_until - self.clock(), 0.0)

    def begin(self, name: str) -> None:
        with self._lock:
            h = self._get(name)
            self._refresh(h)
            if h.state == "half_open":
                h.probe_in_flight = True

    def success(self, name: str, latency_ms: float) -> None:
        with self._lock:
            h = self._get(name)
            h.successes += 1
            if h.state != "closed":
                h.state, h.cooldown_s = "closed", 0.0
            h.failures, h.probe_in_flight, h.opened_at = 0, False, None
            h.latency_ms = latency_ms if h.latency_ms is None else 0.7 * h.latency_ms + 0.3 * latency_ms
            self._remember(h, True)

    def failure(self, name: str, kind: str, summary: str, trips: bool, retry_after: float | None = None,
                immediate: bool = False) -> None:
        with self._lock:
            h = self._get(name)
            now = self.clock()
            h.errors += 1
            h.last_error, h.last_error_at = summary, now
            self._remember(h, False)
            if kind == "rate_limited":
                h.rate_limited_until = max(h.rate_limited_until, now + (retry_after if retry_after else 10.0))
            if not trips:
                h.probe_in_flight = False
                return
            h.failures += 1
            if h.state == "half_open":
                self._open(h, min(max(h.cooldown_s, self.base_cooldown) * 2, self.max_cooldown))
            elif h.state == "closed" and (immediate or h.failures >= self.threshold):
                self._open(h, self.base_cooldown)

    def _open(self, h: DeploymentHealth, cooldown: float) -> None:
        h.state, h.opened_at, h.cooldown_s, h.probe_in_flight = "open", self.clock(), cooldown, False
        h.trips += 1

    def _remember(self, h: DeploymentHealth, ok: bool) -> None:
        h.history.append((self.clock(), ok))
        del h.history[:-60]

    def reset(self, name: str | None = None) -> None:
        with self._lock:
            if name:
                self._h.pop(name, None)
            else:
                self._h.clear()

    def latency(self, name: str) -> float | None:
        with self._lock:
            return self._get(name).latency_ms

    def snapshot(self, names: list[str]) -> dict[str, dict]:
        with self._lock:
            now = self.clock()
            out = {}
            for n in names:
                h = self._get(n)
                self._refresh(h)
                out[n] = {
                    "state": h.state, "consecutive_failures": h.failures,
                    "reopens_in_s": round(h.cooldown_s - (now - h.opened_at), 1) if h.state == "open" else None,
                    "rate_limited_for_s": round(h.rate_limited_until - now, 1) if h.rate_limited_until > now else None,
                    "latency_ms": round(h.latency_ms, 1) if h.latency_ms is not None else None,
                    "last_error": h.last_error,
                    "last_error_ago_s": round(now - h.last_error_at, 1) if h.last_error_at else None,
                    "successes": h.successes, "errors": h.errors, "trips": h.trips,
                    "recent": [ok for _, ok in h.history[-30:]],
                }
            return out
