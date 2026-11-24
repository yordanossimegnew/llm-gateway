"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(v: str | None, default: bool) -> bool:
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/llm_gateway"
    config_file: Path = PROJECT_ROOT / "gateway.yaml"
    require_keys: bool = False          # true: every /v1 request needs a gateway key (gw-...)
    admin_key: str | None = None        # protects key management and chaos; unset = open (local dev)
    enable_chaos: bool = False          # fault injection endpoints (demos, game days)
    log_bodies: bool = False            # store full prompts/replies in the request log, not just previews
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
            raise ValueError("DATABASE_URL must be a PostgreSQL URL.")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    s = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        config_file=Path(os.getenv("GATEWAY_CONFIG", Settings.config_file)),
        require_keys=_bool(os.getenv("REQUIRE_KEYS"), Settings.require_keys),
        admin_key=os.getenv("GATEWAY_ADMIN_KEY") or None,
        enable_chaos=_bool(os.getenv("ENABLE_CHAOS"), Settings.enable_chaos),
        log_bodies=_bool(os.getenv("LOG_BODIES"), Settings.log_bodies),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(s, **{k: v for k, v in overrides.items() if v is not None})
