"""One gateway request as a LangGraph graph.

    START -> lookup --hit--> finalize -> END
               |miss
               v
             plan -> attempt --ok / refusal kept / stop--> finalize
                      ^   |  \
                      |   |   +--next--> attempt   (fall back to the next deployment)
                      |   +--retry--> backoff --+  (same deployment, after a delay)
                      +-------------------------+

* lookup   - exact-match response cache (routes with cache_ttl_s > 0)
* plan     - orders the route's deployments by its strategy (priority / weighted / latency) and
             skips the ones whose circuit is open, that are cooling down after a 429, or whose
             context window is too small for the request
* attempt  - calls one deployment through its LangChain chat model (streaming or not), classifies
             any failure and decides: done, retry the same deployment, fall back, or stop
* backoff  - exponential backoff with jitter, or the provider's Retry-After when it's short enough
* finalize - cost, cache write, and one row in the request log (+ one per attempt)

Streaming falls back only until the first token has been sent: after that the client already holds
part of an answer from one model, so a mid-stream failure ends the stream with an error instead.

The graph has no checkpointer on purpose: a request lives for seconds, and its durable record is the
request log in PostgreSQL. Checkpointing every step would add a database round trip per hop.
"""

from __future__ import annotations

import hashlib
import json
import operator
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessageChunk
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from ..models.catalog import GatewayConfig, Route
from ..models.providers import (
    Failure,
    InjectedError,
    ModelPool,
    bind,
    call_kwargs,
    classify,
    estimate_tokens,
    finish_reason,
    is_refusal,
    make_failure,
    preview,
    text_of,
    to_langchain,
    tool_calls_openai,
    usage_of,
)
from ..tools.chaos import Chaos, midstream_error
from ..tools.health import HealthBoard
from ..tools.store import Store
from ..utils import get_logger
from ..utils.config import Settings

log = get_logger("graph")


@dataclass
class Components:
    settings: Settings
    config: GatewayConfig
    store: Store
    pool: ModelPool
    health: HealthBoard
    chaos: Chaos
    rng: random.Random = field(default_factory=random.Random)
    sleep: Any = time.sleep


class GatewayState(TypedDict, total=False):
    id: str
    request: dict                 # OpenAI-style: model, messages, tools, tool_choice, max_tokens, temperature, stop, stream
    route: str
    key_id: str | None
    use_cache: bool
    started: float
    cache_key: str | None
    cache_hit: bool
    plan: list[str]
    preferred: str | None         # the strategy's first pick before health checks: anything else is a fallback
    skipped: list[dict]
    idx: int
    tries: int                    # attempts so far on plan[idx]
    attempts: Annotated[list[dict], operator.add]
    decision: str                 # done | retry | next | stop
    wait_s: float
    emitted: bool                 # streaming: tokens already sent to the client
    ttft_ms: float | None
    result: dict | None
    failure: dict | None
    response: dict                # what the API returns (set by finalize)


def cache_key(route: str, req: dict) -> str:
    keep = {k: req.get(k) for k in ("messages", "tools", "tool_choice", "temperature", "top_p", "max_tokens", "stop")}
    return hashlib.sha256(json.dumps({"route": route, **keep}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def build_graph(parts: Components):
    cfg = parts.config

    def route_of(state: GatewayState) -> Route:
        return cfg.resolve(state["route"])

    # ------------------------------------------------------------------ lookup
    def lookup(state: GatewayState) -> dict:
        route = route_of(state)
        if not (route.cache_ttl_s and state.get("use_cache", True)):
            return {"cache_key": None, "cache_hit": False}
        key = cache_key(route.name, state["request"])
        hit = parts.store.cache_get(key)
        if hit:
            if state["request"].get("stream"):
                w = get_stream_writer()
                if hit.get("text"):
                    w({"type": "text", "text": hit["text"]})
                for i, c in enumerate(hit.get("tool_calls") or []):
                    w({"type": "tool_call", "index": i, "id": c["id"], "name": c["function"]["name"],
                       "arguments": c["function"]["arguments"]})
            return {"cache_key": key, "cache_hit": True, "result": {**hit, "usage": {}, "cached": True},
                    "ttft_ms": (time.perf_counter() - state["started"]) * 1000}
        return {"cache_key": key, "cache_hit": False}

    # ------------------------------------------------------------------ plan
    def plan(state: GatewayState) -> dict:
        route = route_of(state)
        req = state["request"]
        names = list(route.deployments)
        if route.strategy == "weighted":
            avail = [n for n in names if parts.health.available(n)[0]]
            if avail:
                first = parts.rng.choices(avail, weights=[max(cfg.deployments[n].weight, 0) or 1 for n in avail])[0]
                names = [first] + [n for n in names if n != first]
        elif route.strategy == "latency":
            names.sort(key=lambda n: parts.health.latency(n) or 0.0)   # unmeasured deployments get tried first

        preferred = names[0]
        need = estimate_tokens(req["messages"], req.get("tools"))
        ordered, skipped = [], []
        for n in names:
            d = cfg.deployments[n]
            if d.context_window and need + (req.get("max_tokens") or d.max_tokens) > d.context_window:
                skipped.append({"deployment": n, "reason": f"context window too small (~{need:,} tokens)"})
                continue
            ok, why = parts.health.available(n)
            if ok:
                ordered.append(n)
            else:
                skipped.append({"deployment": n, "reason": why})
        if not ordered:
            # Everything is unhealthy or cooling down: rather than failing without trying, take the
            # deployment that should recover first (a probe), waiting out a short Retry-After.
            candidates = [s["deployment"] for s in skipped if not s["reason"].startswith("context")]
            if candidates:
                best = min(candidates, key=parts.health.wait_time)
                ordered = [best]
                skipped = [s for s in skipped if s["deployment"] != best] + [
                    {"deployment": best, "reason": "all deployments unavailable: tried anyway as a last resort"}]
        log.info("%s route=%s plan=%s skipped=%s", state["id"], route.name, ordered, [s["deployment"] for s in skipped])
        return {"plan": ordered, "preferred": preferred, "skipped": skipped, "idx": 0, "tries": 0, "emitted": False}

    # ------------------------------------------------------------------ attempt
    def attempt(state: GatewayState) -> dict:
        route = route_of(state)
        req = state["request"]
        plan_ = state["plan"]
        if not plan_:
            f = make_failure("server_error", "no deployment can serve this request")
            f.kind = "unavailable"
            return {"decision": "stop", "failure": _fd(f)}
        name = plan_[state["idx"]]
        d = cfg.deployments[name]
        kind = cfg.providers[d.provider].kind
        elapsed = time.perf_counter() - state["started"]
        if elapsed > route.deadline_s:
            f = make_failure("timeout", f"route deadline of {route.deadline_s:g}s exceeded")
            return {"decision": "stop", "failure": _fd(f)}

        n = len(state.get("attempts") or []) + 1
        started_at = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        emitted = state.get("emitted", False)
        ttft = state.get("ttft_ms")
        parts.health.begin(name)
        try:
            break_after = parts.chaos.before_call(name)
            model = bind(parts.pool.get(name), kind, req.get("tools"), req.get("tool_choice"))
            msgs = to_langchain(req["messages"])
            kw = call_kwargs(kind, d, req)
            if req.get("stream"):
                writer = get_stream_writer()
                full: AIMessageChunk | None = None
                chunks, idx_map = 0, {}
                for chunk in model.stream(msgs, **kw):
                    full = chunk if full is None else full + chunk
                    events = []
                    if t := text_of(chunk):
                        events.append({"type": "text", "text": t})
                    for tc in getattr(chunk, "tool_call_chunks", None) or []:
                        i = idx_map.setdefault(tc.get("index", len(idx_map)), len(idx_map))
                        events.append({"type": "tool_call", "index": i, "id": tc.get("id"), "name": tc.get("name"),
                                       "arguments": tc.get("args") or ""})
                    if not events:
                        continue
                    if break_after is not None and chunks >= break_after:
                        raise midstream_error()
                    if not emitted:
                        emitted = True
                        ttft = (time.perf_counter() - state["started"]) * 1000
                        writer({"type": "start", "deployment": name, "model": d.model})
                    for e in events:
                        writer(e)
                    chunks += 1
                msg = full if full is not None else AIMessageChunk(content="")
            else:
                msg = model.invoke(msgs, **kw)
            latency = (time.perf_counter() - t0) * 1000
            if is_refusal(msg) and route.fallback_on_refusal and not emitted:
                raise InjectedError(make_failure("refusal", "the model declined to answer", 200))
        except Exception as exc:  # noqa: BLE001 - every provider error is classified, none escapes
            latency = (time.perf_counter() - t0) * 1000
            f = exc.failure if isinstance(exc, InjectedError) else classify(exc)
            if not isinstance(exc, InjectedError):
                log.warning("%s %s failed: %s (%s)", state["id"], name, f.summary(), type(exc).__name__)
            parts.health.failure(name, f.kind, f.summary(), f.trips_circuit, f.retry_after,
                                 immediate=f.kind in ("auth", "not_found"))
            rec = _attempt(n, name, started_at, latency, f)
            decision, wait = _decide(route, state, f, emitted, len(plan_), name)
            return {"attempts": [rec], "failure": _fd(f), "decision": decision, "wait_s": wait,
                    "emitted": emitted, "ttft_ms": ttft}

        parts.health.success(name, latency)
        result = {"deployment": name, "model": d.model, "text": text_of(msg), "tool_calls": tool_calls_openai(msg),
                  "finish_reason": finish_reason(msg), "usage": usage_of(msg)}
        return {"attempts": [_attempt(n, name, started_at, latency, None)], "result": result, "decision": "done",
                "failure": None, "emitted": emitted, "ttft_ms": ttft}

    def _decide(route: Route, state: GatewayState, f: Failure, emitted: bool, n_plan: int,
                name: str) -> tuple[str, float]:
        if emitted:
            return "stop", 0.0          # the client already has part of an answer
        tries = state.get("tries", 0)
        left = route.deadline_s - (time.perf_counter() - state["started"])
        has_next = state["idx"] + 1 < n_plan
        if f.retry and tries < route.retries and parts.health.available(name)[0]:   # not if that failure opened the circuit
            wait = route.backoff_s * (2 ** tries) * (0.5 + parts.rng.random())
            if f.retry_after is not None and f.retry_after <= route.max_wait_s:
                wait = max(wait, f.retry_after)
            if wait < left:
                return "retry", wait
        if f.fallback and has_next:
            return "next", 0.0
        if f.kind == "rate_limited" and tries < route.retries and f.retry_after is not None \
                and f.retry_after <= min(route.max_wait_s, left):
            return "retry", f.retry_after     # nothing left to fall back to and the wait is short
        return "stop", 0.0

    def backoff(state: GatewayState) -> dict:
        parts.sleep(state.get("wait_s", 0.0))
        return {"tries": state.get("tries", 0) + 1}

    def advance(state: GatewayState) -> dict:
        return {"idx": state["idx"] + 1, "tries": 0}

    # ------------------------------------------------------------------ finalize
    def finalize(state: GatewayState) -> dict:
        route = route_of(state)
        req = state["request"]
        res = state.get("result")
        attempts = state.get("attempts") or []
        f = state.get("failure")
        latency = (time.perf_counter() - state["started"]) * 1000
        first_choice = state.get("preferred") or route.deployments[0]
        if res:
            d = cfg.deployments.get(res["deployment"])
            usage = res.get("usage") or {}
            cost = 0.0 if res.get("cached") else (d.price.cost(usage) if d else 0.0)
            if state.get("cache_key") and not res.get("cached") and res["finish_reason"] in ("stop", "tool_calls"):
                parts.store.cache_put(state["cache_key"], route.name,
                                      {k: res[k] for k in ("deployment", "model", "text", "tool_calls", "finish_reason")},
                                      route.cache_ttl_s)
            status, http = "ok", 200
        else:
            usage, cost = {}, 0.0
            status, http = "error", _http_status(f, attempts)
            if http == 429 and f:   # tell the client when the first deployment frees up
                waits = [parts.health.wait_time(a["deployment"]) for a in attempts]
                f = {**f, "retry_after": round(min(waits), 1) if waits else f.get("retry_after")}
        failed_mid = bool(res is None and state.get("emitted"))
        row = {
            "id": state["id"], "created_at": datetime.now(timezone.utc), "key_id": state.get("key_id"),
            "route": route.name, "served_by": res["deployment"] if res else None, "model": res["model"] if res else None,
            "status": status, "http_status": http,
            "error_kind": None if res else (f or {}).get("kind"),
            "error": None if res else (("stream interrupted after the first token: " if failed_mid else "")
                                       + (f or {}).get("summary", "")),
            "stream": bool(req.get("stream")), "cache_hit": bool(state.get("cache_hit")),
            "fallback": bool(res and not res.get("cached") and res["deployment"] != first_choice),
            "attempts": len(attempts), "latency_ms": round(latency, 1),
            "ttft_ms": round(state["ttft_ms"], 1) if state.get("ttft_ms") is not None else None,
            "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
            "cost_usd": round(cost, 8), "finish_reason": res["finish_reason"] if res else None,
            "prompt": json.dumps(req["messages"], ensure_ascii=False)[:20000] if parts.settings.log_bodies
            else preview(req["messages"]),
            "reply": (res["text"] if parts.settings.log_bodies else (res["text"] or "")[:300]) if res else None,
            "skipped": state.get("skipped") or [],
        }
        try:
            parts.store.log_request(row, attempts)
        except Exception:  # noqa: BLE001 - never fail a served request because the log write failed
            log.exception("%s: could not write the request log", state["id"])
        return {"response": {**row, "created_at": row["created_at"].isoformat(), "result": res, "failure": f,
                             "trace": attempts, "usage": usage}}

    g = StateGraph(GatewayState)
    g.add_node("lookup", lookup)
    g.add_node("plan", plan)
    g.add_node("attempt", attempt)
    g.add_node("backoff", backoff)
    g.add_node("advance", advance)
    g.add_node("finalize", finalize)
    g.add_edge(START, "lookup")
    g.add_conditional_edges("lookup", lambda s: "finalize" if s.get("cache_hit") else "plan", ["finalize", "plan"])
    g.add_edge("plan", "attempt")
    g.add_conditional_edges("attempt", lambda s: {"retry": "backoff", "next": "advance"}.get(s["decision"], "finalize"),
                            ["backoff", "advance", "finalize"])
    g.add_edge("backoff", "attempt")
    g.add_edge("advance", "attempt")
    g.add_edge("finalize", END)
    return g.compile()


def _fd(f: Failure) -> dict:
    return {"kind": f.kind, "message": f.message, "status": f.status, "retry_after": f.retry_after,
            "injected": f.injected, "summary": f.summary()}


def _attempt(n: int, name: str, started_at: datetime, latency: float, f: Failure | None) -> dict:
    return {"n": n, "deployment": name, "outcome": "error" if f else "ok", "error_kind": f.kind if f else None,
            "status_code": f.status if f else 200, "detail": f.summary() if f else None,
            "injected": bool(f and f.injected), "latency_ms": round(latency, 1), "started_at": started_at}


def _http_status(f: dict | None, attempts: list[dict]) -> int:
    """What the gateway answers when every option failed."""
    if not f:
        return 502
    kind = f.get("kind")
    if kind == "bad_request":
        return f.get("status") if (f.get("status") or 0) in range(400, 500) else 400
    if kind == "context_length":
        return 400
    if kind == "refusal":
        return 422
    if attempts and all(a["error_kind"] == "rate_limited" for a in attempts):
        return 429
    if kind == "timeout":
        return 504
    return 503 if kind in ("unavailable", "overloaded") else 502


def new_request_id() -> str:
    return "gwr_" + uuid.uuid4().hex[:16]
