"""The HTTP API, driven by the official OpenAI Python SDK where it matters (that's what clients will use)."""

from __future__ import annotations

import json
from dataclasses import replace

import openai
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from tests.test_routing import WEATHER_TOOL


@pytest.fixture
def app_url(gw, settings, serve):
    return serve(create_app(settings, gateway=gw))


@pytest.fixture
def client(gw, settings):
    with TestClient(create_app(settings, gateway=gw)) as c:
        yield c


def sdk(url: str, key: str = "anything") -> openai.OpenAI:
    return openai.OpenAI(base_url=f"{url}/v1", api_key=key, max_retries=0)


def test_openai_sdk_completion_with_gateway_metadata(app_url, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    raw = sdk(app_url).chat.completions.with_raw_response.create(
        model="smart", messages=[{"role": "user", "content": "write a haiku"}])
    r = raw.parse()
    assert r.choices[0].message.content.startswith("First choice overloaded")
    assert r.model == "claude-sonnet-5-5" and r.usage.completion_tokens > 0
    assert raw.headers["x-gateway-served-by"] == "sonnet" and raw.headers["x-gateway-fallback"] == "true"
    meta = r.model_extra["x_gateway"]
    assert [t["error_kind"] for t in meta["trace"]] == ["overloaded", "overloaded", None]


def test_openai_sdk_streaming(app_url, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    stream = sdk(app_url).chat.completions.create(
        model="smart", messages=[{"role": "user", "content": "explain backoff"}], stream=True,
        stream_options={"include_usage": True})
    chunks = list(stream)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.startswith("When a call fails")
    assert chunks[0].model == "claude-sonnet-5-5"
    finish = [c for c in chunks if c.choices and c.choices[0].finish_reason]
    assert finish[0].choices[0].finish_reason == "stop" and finish[0].model_extra["x_gateway"]["served_by"] == "sonnet"
    assert chunks[-1].usage.completion_tokens > 0


def test_openai_sdk_streamed_tool_call(app_url, upstream):
    upstream.behave("claude-opus-5-5", tool_call={"name": "get_weather", "args": {"city": "Accra"}})
    chunks = list(sdk(app_url).chat.completions.create(
        model="smart", messages=[{"role": "user", "content": "weather?"}], tools=[WEATHER_TOOL], stream=True))
    calls = [tc for c in chunks if c.choices and c.choices[0].delta.tool_calls for tc in c.choices[0].delta.tool_calls]
    assert calls[0].function.name == "get_weather" and calls[0].id
    assert json.loads("".join(tc.function.arguments or "" for tc in calls)) == {"city": "Accra"}


def test_failure_before_first_token_is_a_real_http_error(app_url, upstream):
    upstream.behave("claude-opus-5-5", script=[400])
    with pytest.raises(openai.BadRequestError):
        sdk(app_url).chat.completions.create(model="smart", messages=[{"role": "user", "content": "hi"}], stream=True)


def test_all_rate_limited_is_429_with_retry_after(client, gw, upstream):
    for d in ("haiku", "sonnet", "local"):
        gw.parts.health.failure(d, "rate_limited", "429", trips=False, retry_after=60)
    for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "llama3.2:3b"):
        upstream.behave(m, script=[429, 429], retry_after=60)
    r = client.post("/v1/chat/completions", json={"model": "fast", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["type"] == "rate_limited"


def test_unknown_model_lists_routes(client):
    r = client.post("/v1/chat/completions", json={"model": "gpt-nope", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404 and "smart" in r.json()["error"]["message"]


def test_models_lists_routes_and_deployments(client):
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert {"smart", "fast", "self-hosted", "opus", "local"} <= set(ids)


def test_keys_budget_rpm_and_route_scope(client, gw, upstream):
    k = client.post("/keys", json={"name": "team-a", "monthly_budget_usd": 0.0002, "rpm": 3, "routes": ["smart"]}).json()
    secret = k["key"]
    assert secret.startswith("gw-") and "key_hash" not in k
    h = {"authorization": f"Bearer {secret}"}
    body = {"model": "smart", "messages": [{"role": "user", "content": "write a haiku"}]}
    assert client.post("/v1/chat/completions", json=body, headers=h).status_code == 200
    r = client.post("/v1/chat/completions", json=body, headers=h)            # first call cost > $0.0002
    assert r.status_code == 429 and r.json()["error"]["code"] == "budget_exceeded"
    client.patch(f"/keys/{k['id']}", json={"monthly_budget_usd": 100})
    assert client.post("/v1/chat/completions", json=body, headers=h).status_code == 200
    r = client.post("/v1/chat/completions", json=body, headers=h)            # 4th request this minute
    assert r.status_code == 429 and r.json()["error"]["code"] == "key_rate_limited"
    r = client.post("/v1/chat/completions", json={**body, "model": "fast"}, headers=h)
    assert r.status_code == 403
    listed = client.get("/keys").json()[0]
    assert listed["requests_this_month"] == 2 and listed["spent_this_month_usd"] > 0
    assert client.post("/v1/chat/completions", json=body,
                       headers={"authorization": "Bearer gw-wrong"}).status_code == 401
    client.patch(f"/keys/{k['id']}", json={"disabled": True})
    assert client.post("/v1/chat/completions", json=body, headers=h).status_code == 403


def test_keys_required_and_admin_key(gw, settings):
    strict = replace(settings, require_keys=True, admin_key="s3cret")
    with TestClient(create_app(strict, gateway=gw)) as c:
        body = {"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
        assert c.post("/v1/chat/completions", json=body).status_code == 401
        assert c.post("/keys", json={"name": "x"}).status_code == 401
        key = c.post("/keys", json={"name": "x"}, headers={"x-admin-key": "s3cret"}).json()["key"]
        assert c.post("/v1/chat/completions", json=body, headers={"authorization": f"Bearer {key}"}).status_code == 200


def test_request_log_detail_and_stats(client, upstream):
    upstream.behave("claude-opus-5-5", script=[529, 529])
    rid = client.post("/v1/chat/completions", json={"model": "smart", "messages": [{"role": "user", "content": "hi"}]}
                      ).headers["x-gateway-request-id"]
    detail = client.get(f"/requests/{rid}").json()
    assert detail["served_by"] == "sonnet" and [a["deployment"] for a in detail["trace"]] == ["opus", "opus", "sonnet"]
    assert client.get("/requests", params={"fallback": True}).json()[0]["id"] == rid
    s = client.get("/stats", params={"minutes": 5}).json()
    assert s["totals"]["rescued"] == 1
    deps = {d["name"]: d for d in client.get("/deployments").json()}
    assert deps["opus"]["health"]["errors"] == 2 and deps["sonnet"]["health"]["latency_ms"] is not None


def test_chaos_endpoints(client, gw, settings):
    r = client.post("/chaos", json={"deployment": "opus", "kind": "overloaded", "duration_s": 60})
    assert r.status_code == 201 and client.get("/chaos").json()["faults"]["opus"]["kind"] == "overloaded"
    out = client.post("/v1/chat/completions", json={"model": "smart", "messages": [{"role": "user", "content": "hi"}]})
    assert out.json()["x_gateway"]["trace"][0]["injected"] is True
    assert client.delete("/chaos/opus").status_code == 204 and client.get("/chaos").json()["faults"] == {}
    assert client.post("/chaos", json={"deployment": "opus", "kind": "nonsense"}).status_code == 422
    with TestClient(create_app(replace(settings, enable_chaos=False), gateway=gw)) as c:
        assert c.post("/chaos", json={"deployment": "opus", "kind": "overloaded"}).status_code == 403
