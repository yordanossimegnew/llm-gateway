"""FastAPI application: an OpenAI-compatible /v1/chat/completions in front of every provider, plus admin endpoints."""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agent import Gateway, build_gateway
from ..utils import get_logger, setup_logging
from ..utils.config import Settings, load_settings

log = get_logger("api")


class ChatRequest(BaseModel):
    """The OpenAI chat completions request (the fields the gateway understands; others are ignored)."""
    model_config = ConfigDict(extra="allow")
    model: str = Field(description="A route (e.g. 'smart') or a deployment name (no fallback).")
    messages: list[dict] = Field(min_length=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: dict | None = None
    tools: list[dict] | None = None
    tool_choice: Any = None
    user: str | None = None

    def to_request(self) -> dict:
        return {"model": self.model, "messages": self.messages,
                "max_tokens": self.max_completion_tokens or self.max_tokens, "temperature": self.temperature,
                "top_p": self.top_p, "stop": self.stop, "tools": self.tools, "tool_choice": self.tool_choice}


class KeyIn(BaseModel):
    name: str = Field(min_length=1)
    monthly_budget_usd: float | None = Field(None, gt=0)
    rpm: int | None = Field(None, ge=1)
    routes: list[str] | None = None


class KeyPatch(BaseModel):
    monthly_budget_usd: float | None = None
    rpm: int | None = None
    routes: list[str] | None = None
    disabled: bool | None = None


class ChaosIn(BaseModel):
    deployment: str
    kind: str = Field(description="overloaded | rate_limited | server_error | timeout | auth | refusal | midstream | latency")
    rate: float = Field(1.0, ge=0, le=1)
    latency_ms: float = Field(0, ge=0)
    duration_s: float | None = Field(None, gt=0)
    retry_after_s: float = Field(5, ge=0)
    after_chunks: int = Field(3, ge=0)


def oai_error(status: int, message: str, kind: str, code: str | None = None, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": kind, "code": code or kind}}, status, headers=headers)


def gateway_meta(r: dict) -> dict:
    return {"request_id": r["id"], "route": r["route"], "served_by": r["served_by"], "attempts": r["attempts"],
            "fallback": r["fallback"], "cache_hit": r["cache_hit"], "cost_usd": r["cost_usd"],
            "latency_ms": r["latency_ms"], "skipped": r["skipped"],
            "trace": [{k: a[k] for k in ("n", "deployment", "outcome", "error_kind", "status_code", "detail", "injected",
                                         "latency_ms")} for a in r["trace"]]}


def gateway_headers(r: dict) -> dict:
    return {"x-gateway-request-id": r["id"], "x-gateway-route": r["route"], "x-gateway-served-by": r["served_by"] or "",
            "x-gateway-attempts": str(r["attempts"]), "x-gateway-fallback": str(r["fallback"]).lower(),
            "x-gateway-cache": "hit" if r["cache_hit"] else "miss"}


def error_response(r: dict) -> JSONResponse:
    f = r.get("failure") or {}
    headers = gateway_headers(r)
    if r["http_status"] == 429 and f.get("retry_after"):
        headers["retry-after"] = str(max(1, round(f["retry_after"])))
    body = {"error": {"message": f"all deployments failed; last error: {f.get('summary', 'unknown')}",
                      "type": f.get("kind", "gateway_error"), "code": f.get("kind")}, "x_gateway": gateway_meta(r)}
    return JSONResponse(body, r["http_status"], headers=headers)


def completion_body(r: dict) -> dict:
    res = r["result"]
    msg: dict[str, Any] = {"role": "assistant", "content": res["text"] or (None if res["tool_calls"] else "")}
    if res["tool_calls"]:
        msg["tool_calls"] = res["tool_calls"]
    u = r["usage"]
    return {"id": "chatcmpl-" + r["id"][4:], "object": "chat.completion", "created": int(time.time()),
            "model": res["model"], "choices": [{"index": 0, "message": msg, "finish_reason": res["finish_reason"]}],
            "usage": {"prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
                      "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0),
                      "prompt_tokens_details": {"cached_tokens": u.get("cache_read", 0)}},
            "x_gateway": gateway_meta(r)}


def create_app(settings: Settings | None = None, gateway: Gateway | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    holder = {"gw": gateway}
    lock = threading.Lock()

    def gw() -> Gateway:
        with lock:
            if holder["gw"] is None:
                holder["gw"] = build_gateway(settings)
            return holder["gw"]

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = 200  # requests block a thread while upstream works
        gw()
        yield

    api = FastAPI(title="LLM Gateway", version="1.0.0", lifespan=lifespan,
                  description="One OpenAI-compatible endpoint in front of Anthropic, OpenAI-compatible servers and "
                              "Ollama, with fallback chains, retries, circuit breakers, budgets and a request log.")

    def admin(x_admin_key: str | None = Header(None)) -> None:
        if settings.admin_key and x_admin_key != settings.admin_key:
            raise HTTPException(401, "admin key required (x-admin-key header)")

    def caller(authorization: str | None) -> tuple[dict | None, JSONResponse | None]:
        secret = (authorization or "").removeprefix("Bearer ").strip()
        if secret.startswith("gw-"):
            key = gw().parts.store.key_by_secret(secret)
            if not key:
                return None, oai_error(401, "invalid gateway key", "authentication_error", "invalid_api_key")
            if key["disabled"]:
                return None, oai_error(403, "this gateway key is disabled", "permission_error", "key_disabled")
            return key, None
        if settings.require_keys:
            return None, oai_error(401, "a gateway key (Authorization: Bearer gw-...) is required",
                                   "authentication_error", "missing_api_key")
        return None, None

    # ------------------------------------------------------------------ OpenAI-compatible
    @api.post("/v1/chat/completions", tags=["openai"])
    def chat_completions(body: ChatRequest, authorization: str | None = Header(None),
                         x_gateway_cache: str | None = Header(None)):
        key, err = caller(authorization)
        if err:
            return err
        try:
            route = gw().config.resolve(body.model)
        except KeyError:
            return oai_error(404, f"unknown model {body.model!r}: use a route ({', '.join(gw().config.routes)}) "
                                  "or a deployment name", "invalid_request_error", "model_not_found")
        if key:
            if key["routes"] and route.name not in key["routes"]:
                return oai_error(403, f"this key may not use route {route.name!r}", "permission_error", "route_forbidden")
            if wait := gw().limiter.check(key["id"], key["rpm"]):
                return oai_error(429, f"key rate limit of {key['rpm']} requests/minute reached", "rate_limit_error",
                                 "key_rate_limited", {"retry-after": str(max(1, round(wait)))})
            if key["monthly_budget_usd"] is not None:
                spent = gw().parts.store.spent_this_month(key["id"])
                if spent >= key["monthly_budget_usd"]:
                    return oai_error(429, f"monthly budget of ${key['monthly_budget_usd']:.2f} used up "
                                          f"(${spent:.4f} spent)", "insufficient_quota", "budget_exceeded")
        req = body.to_request()
        use_cache = (x_gateway_cache or "").lower() not in ("no", "off", "bypass", "false")
        kid = key["id"] if key else None
        if not body.stream:
            r = gw().complete(req, kid, use_cache)
            if r["status"] != "ok":
                return error_response(r)
            return JSONResponse(completion_body(r), headers=gateway_headers(r))
        return stream_response(gw().stream(req, kid, use_cache), bool((body.stream_options or {}).get("include_usage")))

    def stream_response(events, include_usage: bool):
        # Pull events until the first token (or the end): failures before any output still get a real HTTP status.
        head = []
        for e in events:
            head.append(e)
            if e["type"] != "start":
                break
        last = head[-1]
        if last["type"] == "final" and last["response"]["status"] != "ok":
            return error_response(last["response"])

        def sse():
            cid = "chatcmpl-" + str(int(time.time() * 1000))
            model = None
            sent_role = False
            for e in _chain(head, events):
                t = e["type"]
                if t == "start":
                    model = e["model"]
                    continue
                if t == "final":
                    r = e["response"]
                    model = model or (r["result"] or {}).get("model")
                    base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
                    if r["status"] == "ok":
                        yield _sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason":
                                                         r["result"]["finish_reason"]}], "x_gateway": gateway_meta(r)})
                    else:
                        yield _sse({"error": {"message": r["error"], "type": r["error_kind"], "code": "stream_interrupted"},
                                    "x_gateway": gateway_meta(r)})
                    if include_usage:
                        u = r["usage"]
                        yield _sse({**base, "choices": [], "usage": {
                            "prompt_tokens": u.get("input_tokens", 0), "completion_tokens": u.get("output_tokens", 0),
                            "total_tokens": u.get("input_tokens", 0) + u.get("output_tokens", 0)}})
                    yield "data: [DONE]\n\n"
                    return
                delta: dict[str, Any] = {}
                if not sent_role:
                    delta["role"], sent_role = "assistant", True
                if t == "text":
                    delta["content"] = e["text"]
                else:
                    tc: dict[str, Any] = {"index": e["index"], "function": {"arguments": e["arguments"]}}
                    if e.get("id"):
                        tc["id"], tc["type"] = e["id"], "function"
                    if e.get("name"):
                        tc["function"]["name"] = e["name"]
                    delta["tool_calls"] = [tc]
                yield _sse({"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    @api.get("/v1/models", tags=["openai"])
    def models() -> dict:
        cfg = gw().config
        data = [{"id": n, "object": "model", "owned_by": "gateway-route", "description": r.description,
                 "deployments": r.deployments} for n, r in cfg.routes.items()]
        data += [{"id": n, "object": "model", "owned_by": d.provider, "model": d.model}
                 for n, d in cfg.deployments.items()]
        return {"object": "list", "data": data}

    # ------------------------------------------------------------------ gateway state
    @api.get("/health")
    def health() -> dict:
        deps = gw().deployments()
        return {"status": "ok", "routes": len(gw().config.routes), "deployments": len(deps),
                "open_circuits": [d["name"] for d in deps if d["health"]["state"] != "closed"],
                "chaos_enabled": settings.enable_chaos, "keys_required": settings.require_keys}

    @api.get("/routes")
    def routes() -> list[dict]:
        return gw().routes()

    @api.get("/deployments")
    def deployments() -> list[dict]:
        return gw().deployments()

    @api.post("/deployments/{name}/reset", dependencies=[Depends(admin)])
    def reset(name: str) -> dict:
        if name not in gw().config.deployments:
            raise HTTPException(404, "deployment not found")
        gw().parts.health.reset(name)
        return {"reset": name}

    @api.get("/requests")
    def list_requests(limit: int = Query(100, ge=1, le=1000), route: str | None = None, status: str | None = None,
                      fallback: bool | None = None, key_id: str | None = None) -> list[dict]:
        return gw().parts.store.requests(limit, route, status, fallback, key_id)

    @api.get("/requests/{rid}")
    def get_request(rid: str) -> dict:
        r = gw().parts.store.request(rid)
        if not r:
            raise HTTPException(404, "request not found")
        return r

    @api.get("/stats")
    def stats(minutes: int = Query(60, ge=1, le=60 * 24 * 31)) -> dict:
        return gw().parts.store.stats(minutes)

    # ------------------------------------------------------------------ keys
    @api.post("/keys", status_code=201, dependencies=[Depends(admin)])
    def create_key(body: KeyIn) -> dict:
        unknown = set(body.routes or []) - set(gw().config.routes) - set(gw().config.deployments)
        if unknown:
            raise HTTPException(422, f"unknown routes: {sorted(unknown)}")
        row, secret = gw().parts.store.create_key(body.name, body.monthly_budget_usd, body.rpm, body.routes)
        return {**{k: v for k, v in row.items() if k != "key_hash"}, "key": secret,
                "note": "store this key now: the gateway keeps only its hash"}

    @api.get("/keys", dependencies=[Depends(admin)])
    def list_keys() -> list[dict]:
        return gw().parts.store.keys()

    @api.patch("/keys/{kid}", dependencies=[Depends(admin)])
    def patch_key(kid: str, body: KeyPatch) -> dict:
        row = gw().parts.store.update_key(kid, **body.model_dump(exclude_unset=True))
        if not row:
            raise HTTPException(404, "key not found")
        return {k: v for k, v in row.items() if k != "key_hash"}

    @api.delete("/keys/{kid}", status_code=204, dependencies=[Depends(admin)])
    def delete_key(kid: str):
        if not gw().parts.store.delete_key(kid):
            raise HTTPException(404, "key not found")

    # ------------------------------------------------------------------ chaos and cache
    def chaos_on() -> None:
        if not settings.enable_chaos:
            raise HTTPException(403, "fault injection is off (set ENABLE_CHAOS=true)")

    @api.get("/chaos")
    def chaos_list() -> dict:
        return {"enabled": settings.enable_chaos, "faults": gw().parts.chaos.active()}

    @api.post("/chaos", status_code=201, dependencies=[Depends(admin), Depends(chaos_on)])
    def chaos_set(body: ChaosIn) -> dict:
        if body.deployment not in gw().config.deployments:
            raise HTTPException(404, "deployment not found")
        try:
            f = gw().parts.chaos.set(body.deployment, body.kind, body.rate, body.latency_ms, body.duration_s,
                                     body.retry_after_s, body.after_chunks)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return f.to_dict()

    @api.delete("/chaos/{deployment}", status_code=204, dependencies=[Depends(admin)])
    def chaos_clear(deployment: str):
        gw().parts.chaos.clear(deployment)

    @api.delete("/chaos", status_code=204, dependencies=[Depends(admin)])
    def chaos_clear_all():
        gw().parts.chaos.clear()

    @api.delete("/cache", dependencies=[Depends(admin)])
    def cache_clear() -> dict:
        return {"deleted": gw().parts.store.cache_clear()}

    @api.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/v1/"):
            return oai_error(exc.status_code, str(exc.detail), "invalid_request_error")
        return JSONResponse({"detail": exc.detail}, exc.status_code)

    return api


def _sse(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False, default=str) + "\n\n"


def _chain(head: list, rest):
    yield from head
    yield from rest
