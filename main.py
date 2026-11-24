"""LLM Gateway: FastAPI app (`uvicorn main:app`) and CLI.

    python main.py serve                                   # http://localhost:8000 (OpenAI-compatible /v1)
    python main.py routes                                  # routes, deployments and their health
    python main.py chat smart "Why do gateways need circuit breakers?" [--stream]
    python main.py keys create "team-search" --budget 50 --rpm 120 [--route smart]
    python main.py keys list
    python main.py stats --minutes 60
"""

from __future__ import annotations

import argparse
import json
import sys

from src.utils.config import load_settings


def __getattr__(name):  # `uvicorn main:app` builds the app lazily
    if name == "app":
        from src.api import create_app
        return create_app()
    raise AttributeError(name)


def _trace(r: dict) -> str:
    lines = []
    for s in r.get("skipped") or []:
        lines.append(f"  skip  {s['deployment']:12} {s['reason']}")
    for a in r.get("trace") or []:
        mark = "ok  " if a["outcome"] == "ok" else "fail"
        lines.append(f"  {mark}  {a['deployment']:12} {a['latency_ms']:>8.0f} ms  {a['detail'] or ''}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="llm-gateway")
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("routes", help="routes, deployments and health")
    chat = sub.add_parser("chat", help="send one prompt through a route and show how it was served")
    chat.add_argument("route")
    chat.add_argument("prompt")
    chat.add_argument("--stream", action="store_true")
    chat.add_argument("--max-tokens", type=int)
    keys = sub.add_parser("keys")
    ks = keys.add_subparsers(dest="kcmd", required=True)
    kc = ks.add_parser("create")
    kc.add_argument("name")
    kc.add_argument("--budget", type=float, help="monthly budget in USD")
    kc.add_argument("--rpm", type=int, help="requests per minute")
    kc.add_argument("--route", action="append", dest="routes")
    ks.add_parser("list")
    st = sub.add_parser("stats")
    st.add_argument("--minutes", type=int, default=60)
    args = ap.parse_args(argv)
    settings = load_settings()

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("main:app", host=args.host, port=args.port)
        return 0

    from src.agent import build_gateway
    from src.models import ConfigError
    from src.utils import setup_logging
    setup_logging(settings.log_level, settings.log_dir)
    try:
        gw = build_gateway(settings)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.cmd == "routes":
        for r in gw.routes():
            print(f"{r['name']:12} {r['strategy']:9} {' -> '.join(r['deployments'])}   {r['description']}")
        print()
        for d in gw.deployments():
            h = d["health"]
            print(f"{d['name']:12} {d['provider']:10} {d['model']:24} circuit={h['state']:9} "
                  f"in=${d['price']['input']:g} out=${d['price']['output']:g} /Mtok")
        return 0
    if args.cmd == "chat":
        req = {"model": args.route, "messages": [{"role": "user", "content": args.prompt}],
               "max_tokens": args.max_tokens}
        try:
            if args.stream:
                r = None
                for e in gw.stream(req):
                    if e["type"] == "text":
                        print(e["text"], end="", flush=True)
                    elif e["type"] == "final":
                        r = e["response"]
                print()
            else:
                r = gw.complete(req)
                if r["status"] == "ok":
                    print(r["result"]["text"] or json.dumps(r["result"]["tool_calls"], indent=2))
        except KeyError:
            print(f"unknown route or deployment: {args.route}", file=sys.stderr)
            return 2
        print(f"\n— {r['status']} · served by {r['served_by'] or 'nobody'} · {r['attempts']} attempt(s) · "
              f"{r['latency_ms']:.0f} ms · ${r['cost_usd']:.6f}" + (f" · {r['error']}" if r["error"] else ""),
              file=sys.stderr)
        print(_trace(r), file=sys.stderr)
        return 0 if r["status"] == "ok" else 1
    if args.cmd == "keys":
        if args.kcmd == "create":
            row, secret = gw.parts.store.create_key(args.name, args.budget, args.rpm, args.routes)
            print(f"{secret}\n(id {row['id']}; store it now, only its hash is kept)")
        else:
            for k in gw.parts.store.keys():
                budget = f"${k['monthly_budget_usd']:g}" if k["monthly_budget_usd"] else "no budget"
                print(f"{k['id']}  {k['prefix']}…  {k['name']:20} {budget:>10}  spent ${k['spent_this_month_usd']:.4f}"
                      f"  rpm={k['rpm'] or '–'}  routes={','.join(k['routes'] or ['all'])}"
                      + ("  DISABLED" if k["disabled"] else ""))
        return 0
    if args.cmd == "stats":
        s = gw.parts.store.stats(args.minutes)
        t = s["totals"]
        print(f"last {args.minutes} min: {t['requests']} requests, {t['errors']} errors, {t['rescued']} rescued by "
              f"fallback, {t['cache_hits']} cache hits, ${t['cost_usd']:.4f}")
        for d in s["deployments"]:
            print(f"  {d['deployment']:12} attempts={d['attempts']:<5} ok={d['ok']:<5} errors={d['errors']:<4} "
                  f"{json.dumps(d['errors_by_kind'])}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
