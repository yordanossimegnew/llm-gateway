"""Provider access through LangChain chat models, plus one error taxonomy for every provider.

Each deployment gets one cached chat model (ChatAnthropic, ChatOpenAI or ChatOllama) with the
SDK's own retries switched off: the gateway decides when to retry, wait or fall back.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from .catalog import Deployment, Provider

ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"


# ----------------------------------------------------------------------------- errors

@dataclass
class Failure:
    """A classified provider error and what the gateway may do about it."""
    kind: str                      # rate_limited | overloaded | server_error | timeout | connection | auth |
                                   # not_found | context_length | bad_request | refusal | internal
    message: str
    status: int | None = None
    retry_after: float | None = None
    retry: bool = False            # worth another attempt on the same deployment
    fallback: bool = True          # worth trying the next deployment
    trips_circuit: bool = False    # counts toward opening this deployment's circuit
    injected: bool = False         # produced by chaos, not by the provider

    def summary(self) -> str:
        code = f"{self.status} " if self.status else ""
        return f"{code}{self.kind}{' (injected)' if self.injected else ''}: {self.message}"[:300]


POLICY = {  # kind: (retry same deployment, fall back, trips circuit)
    "rate_limited": (False, True, False),   # cooldown instead; waited for only when nothing else is left
    "overloaded": (True, True, True),
    "server_error": (True, True, True),
    "timeout": (False, True, True),
    "connection": (True, True, True),
    "auth": (False, True, True),
    "not_found": (False, True, True),
    "context_length": (False, True, False),  # a model with a bigger window may take it
    "bad_request": (False, False, False),    # the client's request is wrong everywhere
    "refusal": (False, True, False),
    "internal": (False, True, False),        # a bug on our side (not the provider's fault): no retry, no circuit
}
CONTEXT_HINTS = ("prompt is too long", "context length", "context_length", "maximum context", "too many tokens",
                 "context window")


def make_failure(kind: str, message: str, status: int | None = None, retry_after: float | None = None,
                 injected: bool = False) -> Failure:
    retry, fallback, trips = POLICY[kind]
    return Failure(kind, message, status, retry_after, retry, fallback, trips, injected)


def _retry_after(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        if ms := headers.get("retry-after-ms"):
            return float(ms) / 1000
        if v := headers.get("retry-after"):
            try:
                return max(float(v), 0.0)
            except ValueError:
                return max(parsedate_to_datetime(v).timestamp() - time.time(), 0.0)
    except (TypeError, ValueError):
        return None
    return None


def _message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(exc, "error", None) or getattr(exc, "message", None) or exc) or type(exc).__name__


def classify(exc: BaseException) -> Failure:
    """Map an exception from any provider SDK (anthropic, openai, ollama/httpx) to a Failure."""
    if isinstance(exc, InjectedError):
        return exc.failure
    names = {c.__name__ for c in type(exc).__mro__}
    status = getattr(exc, "status_code", None)
    msg = _message(exc)
    if status is None and (names & {"APITimeoutError", "TimeoutException", "ReadTimeout", "TimeoutError",
                                    "ConnectTimeout", "WriteTimeout", "PoolTimeout"}):
        return make_failure("timeout", "provider did not answer in time")
    if status is None and (names & {"APIConnectionError", "ConnectError", "ConnectionError", "RemoteProtocolError",
                                    "ReadError", "NetworkError", "TransportError"}):
        return make_failure("connection", msg)
    if not isinstance(status, int):
        if names & {"APIError", "OpenAIError", "AnthropicError", "ResponseError", "HTTPError"}:
            return make_failure("server_error", f"{type(exc).__name__}: {msg}")
        return make_failure("internal", f"{type(exc).__name__}: {msg}")
    headers = getattr(getattr(exc, "response", None), "headers", None)
    low = msg.lower()
    if status == 429:
        return make_failure("rate_limited", msg, status, _retry_after(headers))
    if status in (529, 503):
        return make_failure("overloaded", msg, status, _retry_after(headers))
    if status >= 500 or status == 408:
        return make_failure("timeout" if status in (408, 504) else "server_error", msg, status)
    if status in (401, 402, 403):
        return make_failure("auth", msg, status)
    if status == 404:
        return make_failure("not_found", msg, status)
    if status == 413 or any(h in low for h in CONTEXT_HINTS):
        return make_failure("context_length", msg, status)
    return make_failure("bad_request", msg, status)


class InjectedError(Exception):
    """Raised by chaos injection; carries a ready-made Failure."""

    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


class RefusalError(Exception):
    """The model declined (HTTP 200, stop_reason refusal / finish_reason content_filter)."""


# ----------------------------------------------------------------------------- messages

def to_langchain(messages: list[dict]) -> list[BaseMessage]:
    """OpenAI chat messages -> LangChain messages (system/developer, user, assistant+tool_calls, tool)."""
    out: list[BaseMessage] = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role in ("system", "developer"):
            out.append(SystemMessage(content=content or ""))
        elif role == "user":
            out.append(HumanMessage(content=content if content is not None else ""))
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"id": c.get("id"), "name": fn.get("name"), "args": args, "type": "tool_call"})
            out.append(AIMessage(content=content or "", tool_calls=calls))
        elif role == "tool":
            out.append(ToolMessage(content=content if content is not None else "", tool_call_id=m.get("tool_call_id", "")))
        else:
            raise ValueError(f"unsupported message role: {role!r}")
    return out


def estimate_tokens(messages: list[dict], tools: list | None = None) -> int:
    """Rough size (4 chars/token) used only to skip deployments whose context window is too small."""
    return len(json.dumps(messages, ensure_ascii=False)) // 4 + (len(json.dumps(tools)) // 4 if tools else 0)


def usage_of(msg: AIMessage) -> dict:
    u = getattr(msg, "usage_metadata", None) or {}
    det = u.get("input_token_details") or {}
    return {"input_tokens": int(u.get("input_tokens") or 0), "output_tokens": int(u.get("output_tokens") or 0),
            "cache_read": int(det.get("cache_read") or 0), "cache_write": int(det.get("cache_creation") or 0)}


def stop_reason(msg: AIMessage) -> str | None:
    md = getattr(msg, "response_metadata", None) or {}
    return md.get("stop_reason") or md.get("finish_reason") or md.get("done_reason")


def is_refusal(msg: AIMessage) -> bool:
    return stop_reason(msg) in ("refusal", "content_filter")


def finish_reason(msg: AIMessage) -> str:
    """Provider stop reasons -> OpenAI finish_reason."""
    if msg.tool_calls:
        return "tool_calls"
    return {"max_tokens": "length", "length": "length", "refusal": "content_filter",
            "content_filter": "content_filter"}.get(stop_reason(msg) or "", "stop")


# ----------------------------------------------------------------------------- chat models

class ModelPool:
    """One LangChain chat model per deployment, built on first use and reused (keeps HTTP connections warm)."""

    def __init__(self, providers: dict[str, Provider], deployments: dict[str, Deployment]):
        self.providers = providers
        self.deployments = deployments
        self._models: dict[str, BaseChatModel] = {}
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Build every model now: the first import of a provider SDK can take seconds, and a user's request
        shouldn't pay for it."""
        for name in self.deployments:
            self.get(name)

    def get(self, name: str) -> BaseChatModel:
        with self._lock:
            if name not in self._models:
                d = self.deployments[name]
                self._models[name] = build_chat_model(self.providers[d.provider], d)
            return self._models[name]


def build_chat_model(p: Provider, d: Deployment) -> BaseChatModel:
    if p.kind == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kw: dict[str, Any] = {}
        extra = dict(d.params)
        if "effort" in extra:
            kw["output_config"] = {"effort": extra.pop("effort")}
        if d.server_fallbacks:
            kw["betas"] = [ANTHROPIC_FALLBACK_BETA]
            extra["fallbacks"] = d.server_fallbacks
        if p.base_url:
            kw["base_url"] = p.base_url
        return ChatAnthropic(model=d.model, max_tokens=d.max_tokens, max_retries=0, timeout=d.timeout_s,
                             api_key=p.api_key() or "missing", model_kwargs=extra, **kw)
    if p.kind == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=d.model, max_retries=0, timeout=d.timeout_s, api_key=p.api_key() or "missing",
                          base_url=p.base_url, stream_usage=True, model_kwargs=dict(d.params))
    from langchain_ollama import ChatOllama
    return ChatOllama(model=d.model, base_url=p.base_url or "http://localhost:11434",
                      client_kwargs={"timeout": d.timeout_s}, **d.params)


def call_kwargs(kind: str, d: Deployment, req: dict) -> dict:
    """Per-request generation parameters in the shape each LangChain model expects."""
    params = {k: req[k] for k in ("temperature", "top_p") if req.get(k) is not None and k not in d.drop_params}
    max_tokens = req.get("max_tokens") or d.max_tokens
    stop = req.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if kind == "ollama":
        opts = {"num_predict": max_tokens, **params}
        if stop:
            opts["stop"] = stop
        return {"options": opts}
    kw: dict[str, Any] = {"max_tokens": max_tokens, **params}
    if stop and "stop" not in d.drop_params:
        kw["stop"] = stop
    return kw


def bind(model: BaseChatModel, kind: str, tools: list | None, tool_choice: Any) -> Any:
    if not tools:
        return model
    choice = None
    if tool_choice not in (None, "auto"):
        if isinstance(tool_choice, dict):
            choice = tool_choice.get("function", {}).get("name")
        else:
            choice = "any" if (tool_choice == "required" and kind == "anthropic") else tool_choice
    return model.bind_tools(tools, tool_choice=choice) if choice else model.bind_tools(tools)


def tool_calls_openai(msg: AIMessage) -> list[dict]:
    return [{"id": c.get("id") or f"call_{i}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False)}}
            for i, c in enumerate(msg.tool_calls or [])]


def text_of(msg: AIMessage) -> str:
    """Visible text only (thinking blocks and tool_use blocks are not part of the reply text)."""
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text") \
        if isinstance(content, list) else ""


_WS = re.compile(r"\s+")


def preview(messages: list[dict], n: int = 160) -> str:
    last = next((m for m in reversed(messages) if m.get("role") == "user"), messages[-1] if messages else {})
    c = last.get("content")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return _WS.sub(" ", str(c or "")).strip()[:n]
