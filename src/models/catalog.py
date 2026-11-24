"""gateway.yaml: providers (where), deployments (which model, at what price) and routes (fallback chains).

    providers:   anthropic / openai / ollama endpoints and credentials
    deployments: one model on one provider, with its prices, timeout and extra parameters
    routes:      what clients ask for (`"model": "smart"`): an ordered chain of deployments,
                 a strategy, retries, a deadline, caching and refusal handling
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

ENV_REF = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


class Provider(BaseModel):
    name: str = ""
    kind: Literal["anthropic", "openai", "ollama"]
    base_url: str | None = None          # None = the provider's public endpoint
    api_key_env: str | None = None       # environment variable holding the key (never the key itself)

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


class Price(BaseModel):
    """USD per million tokens."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float | None = None      # default: 10% of input
    cache_write: float | None = None     # default: 125% of input

    def cost(self, usage: dict) -> float:
        cr = usage.get("cache_read", 0) or 0
        cw = usage.get("cache_write", 0) or 0
        fresh = max((usage.get("input_tokens", 0) or 0) - cr - cw, 0)
        return (fresh * self.input + cr * (self.cache_read if self.cache_read is not None else self.input * 0.1)
                + cw * (self.cache_write if self.cache_write is not None else self.input * 1.25)
                + (usage.get("output_tokens", 0) or 0) * self.output) / 1e6


class Deployment(BaseModel):
    name: str = ""
    provider: str
    model: str
    price: Price = Field(default_factory=Price)
    timeout_s: float = 60.0
    max_tokens: int = 4096                    # used when the client doesn't send one
    context_window: int | None = None         # tokens; requests estimated larger skip this deployment
    params: dict = Field(default_factory=dict)          # extra model kwargs, e.g. {effort: low}
    drop_params: list[str] = Field(default_factory=list)  # client params this model rejects (temperature...)
    server_fallbacks: str | None = None       # Anthropic only: "default" = server-side refusal fallback
    weight: int = 1                           # for the "weighted" strategy


class Route(BaseModel):
    name: str = ""
    description: str = ""
    deployments: list[str] = Field(min_length=1)
    strategy: Literal["priority", "weighted", "latency"] = "priority"
    retries: int = Field(1, ge=0, le=5)       # extra attempts per deployment on transient errors
    backoff_s: float = 0.25                   # first retry delay; doubles, with jitter
    deadline_s: float = 120.0                 # whole request, every attempt included
    max_wait_s: float = 2.0                   # longest Retry-After worth waiting for instead of moving on
    fallback_on_refusal: bool = True
    cache_ttl_s: int = 0                      # >0: identical requests are answered from the cache

    @field_validator("deployments")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("a deployment appears twice in the chain")
        return v


class GatewayConfig(BaseModel):
    providers: dict[str, Provider]
    deployments: dict[str, Deployment]
    routes: dict[str, Route]
    circuit: dict = Field(default_factory=lambda: {"failures": 3, "cooldown_s": 30, "max_cooldown_s": 300})

    @model_validator(mode="after")
    def _link(self) -> "GatewayConfig":
        for n, p in self.providers.items():
            p.name = n
        for n, d in self.deployments.items():
            d.name = n
            if d.provider not in self.providers:
                raise ValueError(f"deployment {n!r}: unknown provider {d.provider!r}")
            if d.server_fallbacks and self.providers[d.provider].kind != "anthropic":
                raise ValueError(f"deployment {n!r}: server_fallbacks is an Anthropic feature")
        for n, r in self.routes.items():
            r.name = n
            if n in self.deployments:
                raise ValueError(f"route {n!r} has the same name as a deployment")
            missing = [d for d in r.deployments if d not in self.deployments]
            if missing:
                raise ValueError(f"route {n!r}: unknown deployments {missing}")
        return self

    def resolve(self, model: str) -> Route:
        """A route by name; a bare deployment name becomes a one-deployment route (no fallback)."""
        if model in self.routes:
            return self.routes[model]
        if model in self.deployments:
            return Route(name=model, deployments=[model], description="direct (no fallback)")
        raise KeyError(model)


def _expand(text: str) -> str:
    """${VAR} and ${VAR:-default} in the YAML, so URLs can differ between local and Docker."""
    return ENV_REF.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), text)


def load_config(path: Path) -> GatewayConfig:
    try:
        raw = yaml.safe_load(_expand(Path(path).read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    try:
        return GatewayConfig.model_validate(raw or {})
    except ValueError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
