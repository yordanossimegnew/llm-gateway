"""Routing behaviour, end to end: real LangChain/SDK clients against the stand-in providers, real PostgreSQL."""

from __future__ import annotations

import json

import pytest

from tests.conftest import ask

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def kinds(r: dict) -> list[tuple[str, str | None]]:
    return [(a["deployment"], a["error_kind"]) for a in r["trace"]]


def test_first_choice_serves_and_cost_is_logged(gw, upstream):
    r = gw.complete(ask("write a haiku"))
    assert r["status"] == "ok" and r["served_by"] == "opus" and not r["fallback"]
    assert r["result"]["text"].startswith("First choice overloaded")
    u = r["usage"]
    assert u["input_tokens"] > 0 and u["output_tokens"] > 0
    assert r["cost_usd"] == pytest.approx((u["input_tokens"] * 4 + u["output_tokens"] * 20) / 1e6)
    row = gw.parts.store.request(r["id"])
    assert row["served_by"] == "opus" and row["cost_usd"] == pytest.approx(r["cost_usd"])
    assert [a["outcome"] for a in row["trace"]] == ["ok"]
    # the Anthropic deployment asks for the server-side refusal fallback and the effort it was configured with
    body = upstream.calls[-1]["body"]
    assert body["fallbacks"] == "default" and body["output_config"] == {"effort": "medium"}


def test_overloaded_retries_once_then_falls_back(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    r = gw.complete(ask())
    assert r["status"] == "ok" and r["served_by"] == "sonnet" and r["fallback"]
    assert kinds(r) == [("opus", "overloaded"), ("opus", "overloaded"), ("sonnet", None)]
    assert len(gw.waits) == 1 and 0 < gw.waits[0] <= 0.25 * 1.5     # one backoff before the retry


def test_transient_error_recovers_on_retry(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[500])
    r = gw.complete(ask())
    assert r["served_by"] == "opus" and not r["fallback"]
    assert kinds(r) == [("opus", "server_error"), ("opus", None)]


def test_rate_limit_moves_on_and_cools_the_deployment_down(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[429], retry_after=7)
    r = gw.complete(ask())
    assert kinds(r) == [("opus", "rate_limited"), ("sonnet", None)]     # no waiting while a fallback exists
    assert gw.waits == []
    before = upstream.count("claude-opus-5-5")
    r2 = gw.complete(ask())
    assert r2["served_by"] == "sonnet" and upstream.count("claude-opus-5-5") == before
    assert r2["skipped"][0]["deployment"] == "opus" and "rate limited" in r2["skipped"][0]["reason"]
    gw.clock.advance(8)
    assert gw.complete(ask())["served_by"] == "opus"


def test_circuit_opens_then_half_open_probe_closes_it(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[500] * 3)
    gw.complete(ask())                    # 2 failures (try + retry)
    gw.complete(ask())                    # 3rd failure opens the circuit
    assert gw.parts.health.snapshot(["opus"])["opus"]["state"] == "open"
    calls = upstream.count("claude-opus-5-5")
    r = gw.complete(ask())
    assert r["served_by"] == "sonnet" and upstream.count("claude-opus-5-5") == calls
    assert "circuit open" in r["skipped"][0]["reason"] and r["fallback"]     # skipping a sick deployment is a fallback
    gw.clock.advance(21)                  # demo config: cooldown 20 s
    r = gw.complete(ask())                # half-open: one probe, it succeeds
    assert r["served_by"] == "opus"
    assert gw.parts.health.snapshot(["opus"])["opus"]["state"] == "closed"


def test_failed_probe_reopens_with_a_longer_cooldown(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[500] * 4)
    gw.complete(ask())
    gw.complete(ask())
    gw.clock.advance(21)
    gw.complete(ask())                    # probe fails
    snap = gw.parts.health.snapshot(["opus"])["opus"]
    assert snap["state"] == "open" and snap["reopens_in_s"] == pytest.approx(40, abs=0.5)


def test_bad_request_is_not_retried_or_fallen_back(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[400])
    r = gw.complete(ask())
    assert r["status"] == "error" and r["http_status"] == 400 and r["error_kind"] == "bad_request"
    assert kinds(r) == [("opus", "bad_request")]


def test_context_length_falls_back_without_retry(gw, upstream):
    upstream.behave("claude-opus-5-5", context_limit=100)
    r = gw.complete(ask("x" * 500))
    assert kinds(r) == [("opus", "context_length"), ("sonnet", None)]
    assert gw.parts.health.snapshot(["opus"])["opus"]["consecutive_failures"] == 0   # not the provider's fault


def test_planner_skips_deployments_with_a_small_context_window(make_gateway, upstream):
    def small(cfg):
        cfg.deployments["opus"].context_window = 1000
    gw = make_gateway(small)
    r = gw.complete(ask("y" * 8000))
    assert r["served_by"] == "sonnet" and upstream.count("claude-opus-5-5") == 0
    assert "context window" in r["skipped"][0]["reason"]


def test_refusal_falls_back_or_is_returned(make_gateway, upstream):
    gw = make_gateway()
    upstream.behave("claude-opus-5-5", script=["refusal"])
    r = gw.complete(ask())
    assert kinds(r) == [("opus", "refusal"), ("sonnet", None)]

    def keep(cfg):
        cfg.routes["smart"].fallback_on_refusal = False
    gw2 = make_gateway(keep)
    upstream.behave("claude-opus-5-5", script=["refusal"])
    r = gw2.complete(ask())
    assert r["status"] == "ok" and r["served_by"] == "opus" and r["result"]["finish_reason"] == "content_filter"


def test_bad_key_opens_the_circuit_at_once(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[401])
    r = gw.complete(ask())
    assert kinds(r) == [("opus", "auth"), ("sonnet", None)]
    assert gw.parts.health.snapshot(["opus"])["opus"]["state"] == "open"


def test_everything_down_gives_503_with_the_full_trace(gw, upstream):
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "qwen2.5-7b-instruct", "llama3.2:3b"):
        upstream.behave(m, fail_rate=1.0, fail_status=503)
    r = gw.complete(ask())
    assert r["status"] == "error" and r["http_status"] == 503
    assert [a["deployment"] for a in r["trace"]] == ["opus", "opus", "sonnet", "sonnet", "qwen-vllm", "qwen-vllm",
                                                     "local", "local"]


def test_all_rate_limited_gives_429_with_retry_after(make_gateway, upstream):
    def no_retry(cfg):
        cfg.routes["fast"].retries = 0
    gw = make_gateway(no_retry)
    for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "llama3.2:3b"):
        upstream.behave(m, script=[429], retry_after=30)
    r = gw.complete(ask(model="fast"))
    assert r["http_status"] == 429
    # Ollama's client drops response headers, so "local" got the default 10 s cooldown: the soonest wins
    assert r["failure"]["retry_after"] == pytest.approx(10, abs=0.5)


def test_short_retry_after_is_waited_out_on_the_last_option(make_gateway, upstream):
    gw = make_gateway()
    upstream.behave("qwen2.5-7b-instruct", script=[429], retry_after=1)
    r = gw.complete(ask(model="qwen-vllm"))      # a deployment name: a one-deployment route
    assert r["status"] == "ok" and kinds(r) == [("qwen-vllm", "rate_limited"), ("qwen-vllm", None)]
    assert gw.waits == [1]


def test_unavailable_everywhere_still_probes_the_soonest(gw, upstream):
    for d in ("opus", "sonnet", "qwen-vllm", "local"):
        gw.parts.health.failure(d, "rate_limited", "429", trips=False, retry_after=60 if d != "local" else 5)
    r = gw.complete(ask())
    assert r["served_by"] == "local"
    assert any("last resort" in s["reason"] for s in r["skipped"])


def test_route_deadline_stops_the_chain(make_gateway, upstream):
    def tight(cfg):
        cfg.routes["smart"].deadline_s = 0.0
    gw = make_gateway(tight)
    r = gw.complete(ask())
    assert r["status"] == "error" and r["http_status"] == 504 and r["attempts"] == 0


def test_timeout_falls_back(make_gateway, upstream):
    def quick(cfg):
        cfg.deployments["opus"].timeout_s = 0.5
    gw = make_gateway(quick)
    upstream.behave("claude-opus-5-5", script=["hang"])
    r = gw.complete(ask())
    assert kinds(r) == [("opus", "timeout"), ("sonnet", None)]
    assert r["trace"][0]["latency_ms"] < 5000


# ----------------------------------------------------------------------------- streaming

def collect(events) -> tuple[str, dict, list[dict]]:
    events = list(events)
    text = "".join(e["text"] for e in events if e["type"] == "text")
    return text, events[-1]["response"], events


def test_stream_falls_back_before_the_first_token(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    text, r, events = collect(gw.stream(ask("explain backoff")))
    assert r["served_by"] == "sonnet" and text.startswith("When a call fails")
    start = next(e for e in events if e["type"] == "start")
    assert start["deployment"] == "sonnet" and start["model"] == "claude-sonnet-5-5"
    assert r["ttft_ms"] is not None and r["usage"]["output_tokens"] > 0


def test_mid_stream_failure_ends_the_stream_without_switching_models(gw, upstream):
    upstream.behave("claude-opus-5-5", script=["drop"])
    text, r, _ = collect(gw.stream(ask("explain backoff")))
    assert text and r["status"] == "error" and "stream interrupted after the first token" in r["error"]
    assert [a["deployment"] for a in r["trace"]] == ["opus"]          # no second model glued onto a partial answer


@pytest.mark.parametrize("deployment", ["qwen-vllm", "local"])
def test_stream_works_for_openai_compatible_and_ollama(gw, upstream, deployment):
    text, r, _ = collect(gw.stream(ask("hello there", model=deployment)))
    assert r["status"] == "ok" and "hello there" in text and r["usage"]["output_tokens"] > 0


# ----------------------------------------------------------------------------- tools

@pytest.mark.parametrize("model", ["opus", "qwen-vllm", "local"])
def test_tool_calls_come_back_in_openai_shape(gw, upstream, model):
    name = gw.config.deployments[model].model
    upstream.behave(name, tool_call={"name": "get_weather", "args": {"city": "Addis Ababa"}})
    r = gw.complete(ask("weather in Addis?", model=model, tools=[WEATHER_TOOL]))
    call = r["result"]["tool_calls"][0]
    assert r["result"]["finish_reason"] == "tool_calls"
    assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Addis Ababa"}
    sent = upstream.calls[-1]["body"]["tools"][0]
    assert (sent.get("name") or sent["function"]["name"]) == "get_weather"


def test_streamed_tool_call_arguments_assemble(gw, upstream):
    upstream.behave("claude-opus-5-5", tool_call={"name": "get_weather", "args": {"city": "Nairobi", "unit": "C"}})
    _, r, events = collect(gw.stream(ask("weather?", tools=[WEATHER_TOOL])))
    parts = [e for e in events if e["type"] == "tool_call"]
    assert parts[0]["name"] == "get_weather" and parts[0]["id"]
    assert json.loads("".join(p["arguments"] for p in parts)) == {"city": "Nairobi", "unit": "C"}
    assert r["result"]["finish_reason"] == "tool_calls"


def test_tool_conversation_round_trips(gw, upstream):
    msgs = [{"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"Lagos\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "31C, sunny"}]
    r = gw.complete({"model": "opus", "messages": msgs, "tools": [WEATHER_TOOL]})
    assert r["status"] == "ok"
    sent = upstream.calls[-1]["body"]["messages"]
    assert sent[1]["content"][-1]["type"] == "tool_use" and sent[2]["content"][0]["type"] == "tool_result"


# ----------------------------------------------------------------------------- strategies, cache, chaos

def test_weighted_strategy_spreads_traffic(make_gateway, upstream):
    def weights(cfg):
        cfg.deployments["qwen-vllm"].weight = 3
        cfg.deployments["local"].weight = 1
        cfg.routes["self-hosted"].cache_ttl_s = 0
    gw = make_gateway(weights)
    results = [gw.complete(ask(f"q{i}", model="self-hosted")) for i in range(80)]
    served = [r["served_by"] for r in results]
    assert 45 <= served.count("qwen-vllm") <= 75 and served.count("local") >= 5
    assert not any(r["fallback"] for r in results)        # a weighted pick is not a fallback


def test_latency_strategy_prefers_the_faster_deployment(make_gateway, upstream):
    def lat(cfg):
        cfg.routes["fast"].strategy = "latency"
    gw = make_gateway(lat)
    gw.parts.health.success("haiku", 900)
    gw.parts.health.success("sonnet", 200)
    gw.parts.health.success("local", 400)
    assert gw.complete(ask(model="fast"))["served_by"] == "sonnet"


def test_cache_answers_identical_requests(gw, upstream):
    r1 = gw.complete(ask("cache me", model="self-hosted"))
    n = upstream.count()
    r2 = gw.complete(ask("cache me", model="self-hosted"))
    assert r2["cache_hit"] and r2["cost_usd"] == 0 and upstream.count() == n
    assert r2["result"]["text"] == r1["result"]["text"]
    r3 = gw.complete(ask("cache me", model="self-hosted"), use_cache=False)
    assert not r3["cache_hit"] and upstream.count() == n + 1
    text, r4, _ = collect(gw.stream(ask("cache me", model="self-hosted")))
    assert r4["cache_hit"] and text == r1["result"]["text"]


def test_chaos_injection_goes_through_the_normal_failure_path(gw, upstream):
    gw.parts.chaos.set("opus", "overloaded")
    r = gw.complete(ask())
    assert r["served_by"] == "sonnet" and upstream.count("claude-opus-5-5") == 0
    assert all(a["injected"] for a in r["trace"][:2]) and "(injected)" in r["trace"][0]["detail"]
    gw.parts.chaos.clear("opus")
    gw.parts.health.reset("opus")
    assert gw.complete(ask())["served_by"] == "opus"


def test_chaos_midstream_break(gw, upstream):
    gw.parts.chaos.set("opus", "midstream", after_chunks=2)
    text, r, _ = collect(gw.stream(ask("explain backoff")))
    assert r["status"] == "error" and len(text.split()) == 2


def test_stats_summarise_the_log(gw, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    gw.complete(ask())
    gw.complete(ask())
    upstream.behave("claude-opus-5-5", script=[400])
    gw.complete(ask())
    s = gw.parts.store.stats(60)
    assert s["totals"]["requests"] == 3 and s["totals"]["rescued"] == 1 and s["totals"]["errors"] == 1
    opus = next(d for d in s["deployments"] if d["deployment"] == "opus")
    assert opus["errors_by_kind"] == {"overloaded": 2, "bad_request": 1} and opus["ok"] == 1
    assert sum(t["direct"] + t["rescued"] + t["failed"] for t in s["timeline"]) == 3
