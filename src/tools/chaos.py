"""Fault injection: make a deployment fail on purpose to see the fallbacks work (demos, game days).

Only active when ENABLE_CHAOS=true. Injected failures go through exactly the same path as real ones
(classification, retries, circuit breaker, fallback) and are labelled "(injected)" in every trace.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import asdict, dataclass

from ..models.providers import Failure, InjectedError, make_failure

FAULTS = {
    # kind: (HTTP status, message)
    "overloaded": (529, "Overloaded"),
    "rate_limited": (429, "Rate limit exceeded"),
    "server_error": (500, "Internal server error"),
    "timeout": (None, "provider did not answer in time"),
    "auth": (401, "invalid x-api-key"),
    "refusal": (200, "the model declined to answer"),
    "midstream": (None, "connection dropped mid-stream"),
}


@dataclass
class Fault:
    deployment: str
    kind: str
    rate: float = 1.0                 # share of calls that fail
    latency_ms: float = 0.0           # added before every call (also when it doesn't fail)
    retry_after_s: float = 5.0        # for rate_limited
    after_chunks: int = 3             # for midstream: fail after this many streamed chunks
    expires_at: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["expires_in_s"] = round(self.expires_at - time.time(), 1) if self.expires_at else None
        return d


class Chaos:
    def __init__(self, enabled: bool = False, seed: int | None = None):
        self.enabled = enabled
        self._faults: dict[str, Fault] = {}
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    def set(self, deployment: str, kind: str, rate: float = 1.0, latency_ms: float = 0.0, duration_s: float | None = None,
            retry_after_s: float = 5.0, after_chunks: int = 3) -> Fault:
        if kind not in FAULTS and kind != "latency":
            raise ValueError(f"unknown fault kind {kind!r}; one of {sorted(FAULTS) + ['latency']}")
        f = Fault(deployment, kind, max(0.0, min(rate, 1.0)), latency_ms, retry_after_s, after_chunks,
                  time.time() + duration_s if duration_s else None)
        with self._lock:
            self._faults[deployment] = f
        return f

    def clear(self, deployment: str | None = None) -> None:
        with self._lock:
            if deployment:
                self._faults.pop(deployment, None)
            else:
                self._faults.clear()

    def active(self) -> dict[str, dict]:
        with self._lock:
            now = time.time()
            for k in [k for k, f in self._faults.items() if f.expires_at and f.expires_at < now]:
                del self._faults[k]
            return {k: f.to_dict() for k, f in self._faults.items()}

    def _fault(self, deployment: str) -> Fault | None:
        if not self.enabled:
            return None
        with self._lock:
            f = self._faults.get(deployment)
            if f and f.expires_at and f.expires_at < time.time():
                del self._faults[deployment]
                return None
            return f

    def before_call(self, deployment: str) -> int | None:
        """Apply latency; raise an injected failure, or return N if the stream should break after N chunks."""
        f = self._fault(deployment)
        if not f:
            return None
        if f.latency_ms:
            time.sleep(f.latency_ms / 1000)
        if f.kind == "latency" or self._rng.random() >= f.rate:
            return None
        if f.kind == "midstream":
            return f.after_chunks
        status, msg = FAULTS[f.kind]
        raise InjectedError(injected(f.kind, msg, status, f.retry_after_s if f.kind == "rate_limited" else None))


def injected(kind: str, msg: str, status: int | None, retry_after: float | None = None) -> Failure:
    base = "connection" if kind == "midstream" else kind
    return make_failure(base, msg, status, retry_after, injected=True)


def midstream_error() -> InjectedError:
    return InjectedError(injected("midstream", FAULTS["midstream"][1], None))
