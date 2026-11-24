"""Pure units: config validation, pricing, error classification, health transitions, limits, message conversion."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.graph import cache_key
from src.models import ConfigError, Price, classify, load_config, to_langchain
from src.models.catalog import Deployment
from src.models.providers import _retry_after, call_kwargs, finish_reason
from src.tools.health import HealthBoard
from src.tools.limits import RateLimiter
from tests.conftest import FakeClock

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_are_valid(monkeypatch):
    monkeypatch.setenv("UPSTREAM_URL", "http://upstream:9100")
    cfg = load_config(ROOT / "gateway.yaml")
    assert cfg.routes["smart"].deployments[0] == "opus" and cfg.deployments["opus"].model == "claude-opus-5-5"
    demo = load_config(ROOT / "demo" / "gateway.yaml")
    assert demo.providers["vllm"].base_url == "http://upstream:9100/v1"


@pytest.mark.parametrize("yaml_text, error", [
    ("providers: {a: {kind: anthropic}}\ndeployments: {d: {provider: b, model: m}}\nroutes: {}", "unknown provider"),
    ("providers: {a: {kind: anthropic}}\ndeployments: {d: {provider: a, model: m}}\n"
     "routes: {r: {deployments: [d, x]}}", "unknown deployments"),
    ("providers: {a: {kind: anthropic}}\ndeployments: {d: {provider: a, model: m}}\n"
     "routes: {r: {deployments: [d, d]}}", "twice"),
    ("providers: {a: {kind: ollama}}\ndeployments: {d: {provider: a, model: m, server_fallbacks: default}}\n"
     "routes: {}", "Anthropic feature"),
])
def test_config_errors(tmp_path, yaml_text, error):
    p = tmp_path / "g.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ConfigError, match=error):
        load_config(p)


def test_price_counts_cache_tokens_at_their_own_rates():
    p = Price(input=4, output=20, cache_read=0.2)
    usage = {"input_tokens": 10_000, "output_tokens": 1_000, "cache_read": 8_000, "cache_write": 1_000}
    expected = (1_000 * 4 + 8_000 * 0.2 + 1_000 * 5 + 1_000 * 20) / 1e6
    assert p.cost(usage) == pytest.approx(expected)


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class _StatusErr(Exception):
    def __init__(self, status, msg="boom", headers=None):
        super().__init__(msg)
        self.status_code, self.message, self.response = status, msg, _Resp(headers or {})


class APITimeoutError(Exception):
    pass


class ConnectError(Exception):
    pass


@pytest.mark.parametrize("exc, kind, retry, fallback", [
    (_StatusErr(429, headers={"retry-after": "3"}), "rate_limited", False, True),
    (_StatusErr(529), "overloaded", True, True),
    (_StatusErr(503), "overloaded", True, True),
    (_StatusErr(500), "server_error", True, True),
    (_StatusErr(504), "timeout", False, True),
    (_StatusErr(401), "auth", False, True),
    (_StatusErr(404), "not_found", False, True),
    (_StatusErr(400, "prompt is too long: 300000 tokens"), "context_length", False, True),
    (_StatusErr(400, "This model's maximum context length is 32768 tokens"), "context_length", False, True),
    (_StatusErr(400, "messages: roles must alternate"), "bad_request", False, False),
    (APITimeoutError(), "timeout", False, True),
    (ConnectError("refused"), "connection", True, True),
    (KeyError("oops"), "internal", False, True),
])
def test_classify(exc, kind, retry, fallback):
    f = classify(exc)
    assert (f.kind, f.retry, f.fallback) == (kind, retry, fallback)


def test_retry_after_header_forms():
    assert _retry_after({"retry-after": "2"}) == 2
    assert _retry_after({"retry-after-ms": "1500", "retry-after": "9"}) == 1.5
    assert _retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.0
    assert _retry_after({}) is None


def test_health_board_transitions():
    clock = FakeClock()
    h = HealthBoard(failures=2, cooldown_s=10, max_cooldown_s=25, clock=clock)
    h.failure("a", "server_error", "500", trips=True)
    assert h.available("a")[0]
    h.failure("a", "server_error", "500", trips=True)
    assert not h.available("a")[0]
    clock.advance(10)
    assert h.available("a")[0]                      # half-open: one probe allowed
    h.begin("a")
    assert h.available("a") == (False, "circuit half-open, probe in flight")
    h.failure("a", "server_error", "500", trips=True)
    assert h.snapshot(["a"])["a"]["reopens_in_s"] == 20
    clock.advance(20)
    h.begin("a")
    h.failure("a", "server_error", "500", trips=True)
    assert h.snapshot(["a"])["a"]["reopens_in_s"] == 25    # capped
    clock.advance(25)
    h.begin("a")
    h.success("a", 100)
    assert h.snapshot(["a"])["a"]["state"] == "closed" and h.available("a")[0]


def test_success_resets_the_failure_streak_and_429_never_trips():
    h = HealthBoard(failures=2, clock=FakeClock())
    h.failure("a", "overloaded", "529", trips=True)
    h.success("a", 50)
    h.failure("a", "overloaded", "529", trips=True)
    for _ in range(5):
        h.failure("b", "rate_limited", "429", trips=False, retry_after=1)
    snap = h.snapshot(["a", "b"])
    assert snap["a"]["state"] == "closed" and snap["b"]["state"] == "closed"


def test_rate_limiter_window():
    clock = FakeClock()
    rl = RateLimiter(clock=clock)
    assert [rl.check("k", 2) for _ in range(2)] == [None, None]
    assert rl.check("k", 2) == 60
    clock.advance(30)
    assert rl.check("k", 2) == 30
    clock.advance(30)
    assert rl.check("k", 2) is None and rl.check("other", 2) is None and rl.check("k", None) is None


def test_openai_messages_to_langchain():
    msgs = to_langchain([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                               "function": {"name": "f", "arguments": "{\"x\": 1}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
    ])
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage, ToolMessage]
    assert msgs[2].tool_calls[0]["args"] == {"x": 1} and msgs[3].tool_call_id == "c1"
    with pytest.raises(ValueError):
        to_langchain([{"role": "wizard", "content": "x"}])


def test_call_kwargs_drop_unsupported_params():
    d = Deployment(provider="p", model="m", drop_params=["temperature"], max_tokens=100)
    req = {"temperature": 0.2, "top_p": 0.9, "stop": "END"}
    assert call_kwargs("anthropic", d, req) == {"max_tokens": 100, "top_p": 0.9, "stop": ["END"]}
    assert call_kwargs("ollama", d, req) == {"options": {"num_predict": 100, "top_p": 0.9, "stop": ["END"]}}


def test_finish_reason_mapping():
    assert finish_reason(AIMessage("x", response_metadata={"stop_reason": "max_tokens"})) == "length"
    assert finish_reason(AIMessage("", response_metadata={"stop_reason": "refusal"})) == "content_filter"
    assert finish_reason(AIMessage("x", response_metadata={"done_reason": "stop"})) == "stop"


def test_cache_key_ignores_irrelevant_fields_and_key_order():
    a = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "stream": True, "user": "u1"}
    b = {"temperature": 0, "messages": [{"content": "hi", "role": "user"}], "stream": False}
    assert cache_key("r", a) == cache_key("r", b) != cache_key("other", a)
