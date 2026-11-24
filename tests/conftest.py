from __future__ import annotations

import copy
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from demo.upstream import DEMO_MODELS, Upstream  # noqa: E402
from src.agent import build_gateway  # noqa: E402
from src.models import load_config  # noqa: E402
from src.utils.config import Settings  # noqa: E402

TEST_DB = os.getenv("TEST_DATABASE_URL")


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def upstream(monkeypatch):
    up = Upstream(copy.deepcopy(DEMO_MODELS), time_scale=0).start()
    monkeypatch.setenv("UPSTREAM_URL", up.base)
    monkeypatch.setenv("DEMO_KEY", "test-key")
    yield up
    up.stop()


@pytest.fixture
def settings(tmp_path):
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL not set")
    eng = create_engine(TEST_DB.replace("postgresql://", "postgresql+psycopg://", 1))
    with eng.begin() as c:
        if c.execute(text("SELECT to_regclass('requests')")).scalar():
            c.execute(text("TRUNCATE requests, attempts, cache, api_keys RESTART IDENTITY CASCADE"))
    eng.dispose()
    return Settings(database_url=TEST_DB, config_file=ROOT / "demo" / "gateway.yaml", log_dir=tmp_path,
                    enable_chaos=True)


@pytest.fixture
def config(upstream):
    return load_config(ROOT / "demo" / "gateway.yaml")   # read after UPSTREAM_URL is set


@pytest.fixture
def make_gateway(settings, config):
    """Build a gateway on the demo config; `edit(cfg)` can change the config first. Sleeps are recorded, not slept."""
    def make(edit=None, settings_=None):
        cfg = config.model_copy(deep=True)
        if edit:
            edit(cfg)
        waits: list[float] = []
        gw = build_gateway(settings_ or settings, config=cfg, sleep=waits.append)
        clock = FakeClock()
        gw.parts.health.clock = clock
        gw.waits, gw.clock = waits, clock
        return gw
    return make


@pytest.fixture
def gw(make_gateway):
    return make_gateway()


def ask(text_: str = "hello", model: str = "smart", **kw) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": text_}], **kw}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve():
    """Run a FastAPI app on a real port (so the official OpenAI SDK can talk to it)."""
    import uvicorn
    servers = []

    def start(app) -> str:
        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.02)
        servers.append(server)
        return f"http://127.0.0.1:{port}"
    yield start
    for s in servers:
        s.should_exit = True
