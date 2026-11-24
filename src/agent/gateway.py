"""The gateway facade: wires the components and runs one request through the graph (blocking or streaming)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from ..models.catalog import GatewayConfig, load_config
from ..models.providers import ModelPool
from ..tools.chaos import Chaos
from ..tools.health import HealthBoard
from ..tools.limits import RateLimiter
from ..tools.store import Store
from ..utils.config import Settings
from .graph import Components, GatewayState, build_graph, new_request_id


class Gateway:
    def __init__(self, parts: Components):
        self.parts = parts
        self.limiter = RateLimiter()
        self.graph = build_graph(parts)

    @property
    def config(self) -> GatewayConfig:
        return self.parts.config

    def _state(self, request: dict, key_id: str | None, use_cache: bool) -> GatewayState:
        route = self.config.resolve(request["model"])   # KeyError for an unknown model
        return {"id": new_request_id(), "request": request, "route": route.name, "key_id": key_id,
                "use_cache": use_cache, "started": time.perf_counter(), "attempts": []}

    def complete(self, request: dict, key_id: str | None = None, use_cache: bool = True) -> dict:
        state = self._state({**request, "stream": False}, key_id, use_cache)
        return self.graph.invoke(state)["response"]

    def stream(self, request: dict, key_id: str | None = None, use_cache: bool = True) -> Iterator[dict]:
        """Yields {"type": "start"|"text"|"tool_call", ...} as tokens arrive, then {"type": "final", "response": ...}."""
        state = self._state({**request, "stream": True}, key_id, use_cache)
        final = None
        for mode, data in self.graph.stream(state, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield data
            else:
                final = data
        yield {"type": "final", "response": final["response"]}

    def deployments(self) -> list[dict]:
        cfg = self.config
        health = self.parts.health.snapshot(list(cfg.deployments))
        chaos = self.parts.chaos.active()
        return [{"name": n, "provider": d.provider, "kind": cfg.providers[d.provider].kind, "model": d.model,
                 "price": d.price.model_dump(), "timeout_s": d.timeout_s, "context_window": d.context_window,
                 "routes": [r.name for r in cfg.routes.values() if n in r.deployments],
                 "health": health[n], "chaos": chaos.get(n)} for n, d in cfg.deployments.items()]

    def routes(self) -> list[dict]:
        return [r.model_dump() for r in self.config.routes.values()]


def build_gateway(settings: Settings, config: GatewayConfig | None = None, store: Store | None = None,
                  config_path: Path | None = None, **component_overrides) -> Gateway:
    config = config or load_config(config_path or settings.config_file)
    if store is None:
        store = Store(settings.database_url)
        store.init()
    c = config.circuit
    parts = Components(settings=settings, config=config, store=store,
                       pool=ModelPool(config.providers, config.deployments),
                       health=HealthBoard(c.get("failures", 3), c.get("cooldown_s", 30), c.get("max_cooldown_s", 300)),
                       chaos=Chaos(enabled=settings.enable_chaos))
    for k, v in component_overrides.items():
        setattr(parts, k, v)
    parts.pool.warm()
    return Gateway(parts)
