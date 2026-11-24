"""A scripted stand-in for LLM providers, speaking the real wire protocols, so the gateway's real SDK
code paths (anthropic, openai, ollama clients via LangChain) can be tested and demoed offline:

* Anthropic:         POST /v1/messages              (JSON, or SSE: message_start ... message_stop; tool_use; refusals)
* OpenAI-compatible: POST /v1/chat/completions      (JSON, or SSE chunks + usage chunk; tool calls)
* Ollama:            POST /api/chat                 (JSON, or NDJSON stream)

Every model can be told to fail: a queue of scripted outcomes (consumed one per call: 429 with
Retry-After, 529, 500, "hang", "refusal", "drop" = break the stream after a few tokens), a failure
rate, extra latency, or a context limit. Replies are canned text, not a model.

    python -m demo.upstream --port 9100        # run it standalone (the demo config points here)
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ERRORS = {
    429: ("rate_limit_error", "Number of request tokens has exceeded your per-minute rate limit"),
    500: ("api_error", "Internal server error"),
    503: ("api_error", "Service unavailable"),
    529: ("overloaded_error", "Overloaded"),
    401: ("authentication_error", "invalid x-api-key"),
    400: ("invalid_request_error", "messages: text content blocks must be non-empty"),
}

DEMO_REPLIES = {
    "circuit breaker": "A circuit breaker stops the gateway from hammering a provider that is already failing: after a "
                       "few errors in a row it skips that provider for a cooldown and sends traffic to the next one. "
                       "That keeps latency low for users and gives the struggling provider room to recover.",
    "haiku": "First choice overloaded,\nthe second model answers—\nthe user never knew.",
    "three things": "1. Which model actually served it, and after how many attempts.\n2. Tokens in and out, with "
                    "the cost.\n3. Latency and time to first token.",
    "backoff": "When a call fails, wait before retrying, and double the wait each time: 0.25 s, 0.5 s, 1 s. Add "
               "randomness (jitter) so thousands of clients don't retry in the same instant and knock the server "
               "over again.",
}


@dataclass
class Behavior:
    script: list = field(default_factory=list)   # outcomes consumed one per call: 429, 529, 500, "hang", "refusal", "drop"
    fail_rate: float = 0.0                       # random failures (status in fail_status)
    fail_status: int = 529
    retry_after: float = 1.0
    ttft_ms: float = 120.0
    tokens_per_sec: float = 60.0
    context_limit: int | None = None             # characters; longer prompts get a 400 "prompt is too long"
    reply: str | None = None                     # fixed reply (default: canned or generated)
    tool_call: dict | None = None                # {"name": ..., "args": {...}} when tools are offered


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+\s*|\s+", text) or [""]


def _last_user(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            return str(c or "")
    return ""


class Upstream:
    def __init__(self, models: dict[str, Behavior] | None = None, time_scale: float = 1.0, seed: int = 7):
        self.models: dict[str, Behavior] = models or {}
        self.time_scale = time_scale
        self.calls: list[dict] = []                # (provider, model, outcome) log for tests
        self.lock = threading.Lock()
        self.rng = random.Random(seed)
        self.server: ThreadingHTTPServer | None = None

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def behave(self, model: str, **kw) -> Behavior:
        with self.lock:
            b = self.models.setdefault(model, Behavior())
            for k, v in kw.items():
                setattr(b, k, v)
            return b

    def count(self, model: str | None = None) -> int:
        return sum(1 for c in self.calls if model is None or c["model"] == model)

    def _sleep(self, ms: float) -> None:
        if self.time_scale:
            time.sleep(ms / 1000 * self.time_scale)

    def _outcome(self, model: str, prompt: str) -> object:
        b = self.models[model]
        with self.lock:
            if b.script:
                return b.script.pop(0)
            if b.fail_rate and self.rng.random() < b.fail_rate:
                return b.fail_status
        if b.context_limit and len(prompt) > b.context_limit:
            return "too_long"
        return "ok"

    def reply_for(self, model: str, prompt: str) -> str:
        b = self.models[model]
        if b.reply:
            return b.reply
        low = prompt.lower()
        for k, v in DEMO_REPLIES.items():
            if k in low:
                return v
        return f"[{model}] {prompt[:80]}".strip()

    # ------------------------------------------------------------------ server
    def start(self, port: int = 0) -> "Upstream":
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def handle(self):
                try:
                    super().handle()
                except (ConnectionError, OSError):
                    pass

            def _json(self, code, obj, headers=None):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _stream_start(self, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

            def _chunk(self, data: bytes):
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def _end(self):
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def _drop(self):  # break the connection mid-stream (no terminating chunk)
                self.wfile.flush()
                self.close_connection = True
                try:
                    self.connection.shutdown(2)
                except OSError:
                    pass

            def _error(self, provider: str, status: int, retry_after: float):
                etype, msg = ERRORS.get(status, ("api_error", "error"))
                headers = {"retry-after": f"{retry_after:g}"} if status == 429 else {}
                if provider == "anthropic":
                    return self._json(status, {"type": "error", "error": {"type": etype, "message": msg},
                                               "request_id": "req_" + uuid.uuid4().hex[:20]}, headers)
                if provider == "openai":
                    return self._json(status, {"error": {"message": msg, "type": etype, "code": None}}, headers)
                return self._json(status, {"error": msg}, headers)

            def do_GET(self):  # noqa: N802
                if self.path == "/_control":
                    return self._json(200, {m: b.__dict__ for m, b in up.models.items()})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [{"name": m, "model": m} for m in up.models]})
                self._json(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/_control":       # the demo script flips behaviours at runtime
                    for model, kw in req.items():
                        up.behave(model, **kw)
                    return self._json(200, {"ok": True})
                provider = {"/v1/messages": "anthropic", "/v1/chat/completions": "openai",
                            "/api/chat": "ollama"}.get(self.path)
                if not provider:
                    return self._json(404, {"error": "not found"})
                model = req.get("model")
                if model not in up.models:
                    return self._error(provider, 404, 0) if provider != "anthropic" else self._json(404, {
                        "type": "error", "error": {"type": "not_found_error", "message": f"model: {model}"}})
                b = up.models[model]
                msgs = req.get("messages", [])
                prompt = _last_user(msgs)
                outcome = up._outcome(model, json.dumps(msgs))
                with up.lock:
                    up.calls.append({"provider": provider, "model": model, "outcome": outcome,
                                     "stream": bool(req.get("stream", provider == "ollama")), "body": req})
                if outcome == "hang":          # real time, whatever time_scale says: this is what timeouts are for
                    time.sleep(30)
                    return self._error(provider, 504, 0)
                if outcome == "too_long":
                    msg = "prompt is too long: 250000 tokens > 200000 maximum"
                    if provider == "anthropic":
                        return self._json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
                    return self._json(400, {"error": {"message": msg, "type": "invalid_request_error",
                                                      "code": "context_length_exceeded"}})
                if isinstance(outcome, int):
                    up._sleep(b.ttft_ms / 3)
                    return self._error(provider, outcome, b.retry_after)
                text = "" if outcome == "refusal" else up.reply_for(model, prompt)
                call = b.tool_call if (req.get("tools") and b.tool_call) else None
                if call:
                    text = ""
                n_in = len(json.dumps(msgs)) // 4 + 8
                toks = _tokens(text) if text else []
                n_out = max(len(toks) + (len(json.dumps(call["args"])) // 4 if call else 0), 1)
                stream = req.get("stream", provider == "ollama")
                up._sleep(b.ttft_ms)
                getattr(self, f"_{provider}")(req, model, b, text, toks, call, n_in, n_out, stream, outcome)

            # -------------------------------------------------------------- Anthropic Messages API
            def _anthropic(self, req, model, b, text, toks, call, n_in, n_out, stream, outcome):
                mid = "msg_" + uuid.uuid4().hex[:24]
                stop = "refusal" if outcome == "refusal" else ("tool_use" if call else "end_turn")
                content = []
                if text:
                    content.append({"type": "text", "text": text})
                if call:
                    content.append({"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": call["name"],
                                    "input": call["args"]})
                usage = {"input_tokens": n_in, "output_tokens": n_out, "cache_creation_input_tokens": 0,
                         "cache_read_input_tokens": 0}
                if not stream:
                    up._sleep(1000 * n_out / b.tokens_per_sec)
                    body = {"id": mid, "type": "message", "role": "assistant", "model": model, "content": content,
                            "stop_reason": stop, "stop_sequence": None, "usage": usage}
                    if stop == "refusal":
                        body["stop_details"] = {"type": "refusal", "category": None, "explanation": None}
                    return self._json(200, body)

                def ev(name, obj):
                    self._chunk(f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode())
                self._stream_start("text/event-stream")
                ev("message_start", {"type": "message_start", "message": {
                    "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                    "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}})
                idx = 0
                if text:
                    ev("content_block_start", {"type": "content_block_start", "index": 0,
                                               "content_block": {"type": "text", "text": ""}})
                    for i, t in enumerate(toks):
                        if outcome == "drop" and i == 3:
                            return self._drop()
                        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                                   "delta": {"type": "text_delta", "text": t}})
                        up._sleep(1000 / b.tokens_per_sec)
                    ev("content_block_stop", {"type": "content_block_stop", "index": 0})
                    idx = 1
                if call:
                    ev("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {
                        "type": "tool_use", "id": content[-1]["id"], "name": call["name"], "input": {}}})
                    args = json.dumps(call["args"])
                    for k in range(0, len(args), 6):
                        ev("content_block_delta", {"type": "content_block_delta", "index": idx,
                                                   "delta": {"type": "input_json_delta", "partial_json": args[k:k + 6]}})
                        up._sleep(1000 / b.tokens_per_sec)
                    ev("content_block_stop", {"type": "content_block_stop", "index": idx})
                delta = {"stop_reason": stop, "stop_sequence": None}
                if stop == "refusal":
                    delta["stop_details"] = {"type": "refusal", "category": None, "explanation": None}
                ev("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": n_out}})
                ev("message_stop", {"type": "message_stop"})
                self._end()

            # -------------------------------------------------------------- OpenAI chat completions
            def _openai(self, req, model, b, text, toks, call, n_in, n_out, stream, outcome):
                finish = "content_filter" if outcome == "refusal" else ("tool_calls" if call else "stop")
                calls = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                          "function": {"name": call["name"], "arguments": json.dumps(call["args"])}}] if call else []
                usage = {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}
                if not stream:
                    up._sleep(1000 * n_out / b.tokens_per_sec)
                    msg = {"role": "assistant", "content": text or None}
                    if calls:
                        msg["tool_calls"] = calls
                    return self._json(200, {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion",
                                            "created": int(time.time()), "model": model,
                                            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                            "usage": usage})
                base = {"id": "chatcmpl-demo", "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": model}

                def send(obj):
                    self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self._stream_start("text/event-stream")
                send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                           "finish_reason": None}]})
                for i, t in enumerate(toks):
                    if outcome == "drop" and i == 3:
                        return self._drop()
                    send({**base, "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
                    up._sleep(1000 / b.tokens_per_sec)
                for i, c in enumerate(calls):
                    send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": i, "id": c["id"], "type": "function",
                        "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}]},
                        "finish_reason": None}]})
                send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                if (req.get("stream_options") or {}).get("include_usage"):
                    send({**base, "choices": [], "usage": usage})
                self._chunk(b"data: [DONE]\n\n")
                self._end()

            # -------------------------------------------------------------- Ollama
            def _ollama(self, req, model, b, text, toks, call, n_in, n_out, stream, outcome):
                ts = "2026-11-24T10:00:00Z"
                tool_calls = [{"function": {"name": call["name"], "arguments": call["args"]}}] if call else []
                done = {"model": model, "created_at": ts, "done": True, "done_reason": "stop",
                        "message": {"role": "assistant", "content": ""}, "prompt_eval_count": n_in, "eval_count": n_out,
                        "total_duration": 1, "load_duration": 0, "prompt_eval_duration": 1, "eval_duration": 1}
                if not stream:
                    up._sleep(1000 * n_out / b.tokens_per_sec)
                    return self._json(200, {**done, "message": {"role": "assistant", "content": text,
                                                                "tool_calls": tool_calls}})
                self._stream_start("application/x-ndjson")
                for i, t in enumerate(toks):
                    if outcome == "drop" and i == 3:
                        return self._drop()
                    self._chunk(json.dumps({"model": model, "created_at": ts, "done": False,
                                            "message": {"role": "assistant", "content": t}}).encode() + b"\n")
                    up._sleep(1000 / b.tokens_per_sec)
                if tool_calls:
                    self._chunk(json.dumps({"model": model, "created_at": ts, "done": False, "message": {
                        "role": "assistant", "content": "", "tool_calls": tool_calls}}).encode() + b"\n")
                self._chunk(json.dumps(done).encode() + b"\n")
                self._end()

        self.server = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


DEMO_MODELS = {
    "claude-opus-5-5": Behavior(ttft_ms=650, tokens_per_sec=55),
    "claude-sonnet-5-5": Behavior(ttft_ms=380, tokens_per_sec=85),
    "claude-haiku-5-5": Behavior(ttft_ms=180, tokens_per_sec=160),
    "llama3.2:3b": Behavior(ttft_ms=240, tokens_per_sec=38),
    "qwen2.5-7b-instruct": Behavior(ttft_ms=300, tokens_per_sec=45),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()
    up = Upstream(copy.deepcopy(DEMO_MODELS)).start(args.port)
    print(f"stand-in providers on {up.base}: Anthropic /v1/messages, OpenAI /v1/chat/completions, Ollama /api/chat")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        up.stop()


if __name__ == "__main__":
    main()
