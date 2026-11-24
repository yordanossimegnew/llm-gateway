"""Streamlit UI for the LLM Gateway (talks to the FastAPI app)."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prompts import SAMPLE_PROMPTS  # noqa: E402

API = os.getenv("GATEWAY_API_URL", "http://127.0.0.1:8000")
ADMIN = {"x-admin-key": os.getenv("GATEWAY_ADMIN_KEY", "")} if os.getenv("GATEWAY_ADMIN_KEY") else {}
# Validated categorical slots 1-3 + the reserved status "critical" for failures (always labelled).
BLUE, ORANGE, AQUA, CRITICAL, GOOD, WARN = "#2a78d6", "#eb6834", "#1baf7a", "#d03b3b", "#0ca30c", "#fab219"
INK, MUTED, SURFACE = "#1f2328", "#6b7280", "#fcfcfb"

st.set_page_config(page_title="LLM Gateway", page_icon="🔀", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.reply {font-size: 0.95rem; line-height: 1.5; white-space: pre-wrap; border: 1px solid #e6e6e6; border-radius: 8px;
        padding: 12px 14px; background: #fafafa; min-height: 4rem;}
.trace {display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: .5rem 0 .2rem 0;}
.step {border: 1px solid #e3e3e3; border-radius: 8px; padding: 6px 10px; font-size: .82rem; background: white;}
.step b {font-weight: 600;} .step .why {color: #6b7280; font-size: .75rem;}
.step.ok {border-color: #0ca30c; background: #f1fbf1;} .step.err {border-color: #d03b3b; background: #fdf3f3;}
.step.skip {border-style: dashed; color: #6b7280;}
.arrow {color: #9ca3af;}
.dep {border: 1px solid #e6e6e6; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px; background: white;}
.dep .name {font-weight: 600; font-size: 1.02rem;} .dep .meta {color: #6b7280; font-size: .8rem;}
.strip span {display: inline-block; width: 7px; height: 14px; margin-right: 2px; border-radius: 2px;}
.tile .l {color: #6b7280; font-size: .85rem;} .tile .v {font-size: 1.45rem; font-weight: 600; color: #1f2328;}
.tile .n {color: #6b7280; font-size: .8rem;}
</style>""", unsafe_allow_html=True)


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, headers={**ADMIN, **kw.pop("headers", {})}, **kw)
    except requests.RequestException as exc:
        st.error(f"Gateway not reachable at {API}: {exc}")
        st.stop()
    try:
        data = r.json() if r.content else {}
    except ValueError:
        data = {"detail": r.text}
    if r.status_code >= 400:
        return {"api_error": data.get("detail") or data.get("error", {}).get("message") or r.text, "status": r.status_code}
    return data


def ms(v):
    return "–" if v is None else (f"{v / 1000:.2f} s" if v >= 1000 else f"{v:.0f} ms")


def usd(v):
    return "–" if v is None else (f"${v:.4f}" if v < 1 else f"${v:,.2f}")


def tile(col, label, value, note=""):
    col.markdown(f"<div class='tile'><div class='l'>{label}</div><div class='v'>{html.escape(str(value))}</div>"
                 f"<div class='n'>{note}</div></div>", unsafe_allow_html=True)


def trace_html(meta: dict) -> str:
    steps = []
    for s in meta.get("skipped") or []:
        steps.append(f"<div class='step skip'>⏭ <b>{s['deployment']}</b><div class='why'>{html.escape(s['reason'])}</div></div>")
    for a in meta.get("trace") or []:
        if a["outcome"] == "ok":
            steps.append(f"<div class='step ok'>✓ <b>{a['deployment']}</b> · {ms(a['latency_ms'])}"
                         f"<div class='why'>served</div></div>")
        else:
            why = a["detail"] or a["error_kind"]
            steps.append(f"<div class='step err'>✗ <b>{a['deployment']}</b> · {ms(a['latency_ms'])}"
                         f"<div class='why'>{html.escape(why[:70])}</div></div>")
    return "<div class='trace'>" + "<span class='arrow'>→</span>".join(steps) + "</div>"


routes = api("GET", "/routes")
deployments = api("GET", "/deployments")
health = api("GET", "/health")
with st.sidebar:
    st.title("🔀 LLM Gateway")
    st.caption("One OpenAI-compatible endpoint · fallbacks · retries · circuit breakers")
    st.markdown("**Routes**")
    for r in routes:
        st.markdown(f"**`{r['name']}`** · {r['strategy']}  \n<span style='color:{MUTED};font-size:.8rem'>"
                    f"{' → '.join(r['deployments'])}</span>", unsafe_allow_html=True)
    st.markdown("**Deployments**")
    for d in deployments:
        dot = {"closed": "🟢", "half_open": "🟡", "open": "🔴"}[d["health"]["state"]]
        st.markdown(f"{dot} **{d['name']}** · {d['model']}" + (" · 💥" if d["chaos"] else ""))
    st.caption(f"API: {API}")

tab_play, tab_traffic, tab_deps, tab_keys = st.tabs(["💬 Playground", "📈 Traffic", "🩺 Deployments", "🔑 Keys"])

# ----------------------------------------------------------------------------- playground
with tab_play:
    c1, c2 = st.columns([1, 3])
    route = c1.selectbox("Route", [r["name"] for r in routes] + [d["name"] for d in deployments], key="play_route")
    stream = c1.toggle("Stream", value=True)
    no_cache = c1.toggle("Bypass cache", value=False)
    prompt = c2.text_area("Prompt", SAMPLE_PROMPTS[0], height=110, key="play_prompt")
    go = c2.button("Send", type="primary")
    out = st.empty()
    meta_box = st.empty()
    if go:
        body = {"model": route, "messages": [{"role": "user", "content": prompt}], "stream": stream}
        hdr = {"x-gateway-cache": "bypass"} if no_cache else {}
        meta = None
        if stream:
            text = ""
            out.markdown("<div class='reply'>…</div>", unsafe_allow_html=True)
            with requests.post(f"{API}/v1/chat/completions", json=body, headers=hdr, stream=True, timeout=180) as r:
                if r.status_code >= 400:
                    err = r.json()
                    out.error(err["error"]["message"])
                    meta = err.get("x_gateway")
                else:
                    for line in r.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        chunk = json.loads(line[6:])
                        if "error" in chunk:
                            out.markdown(f"<div class='reply'>{html.escape(text)}</div>", unsafe_allow_html=True)
                            st.error(chunk["error"]["message"])
                            meta = chunk.get("x_gateway")
                            continue
                        if chunk.get("x_gateway"):
                            meta = chunk["x_gateway"]
                        for ch in chunk.get("choices", []):
                            text += ch["delta"].get("content") or ""
                        out.markdown(f"<div class='reply'>{html.escape(text)}▌</div>", unsafe_allow_html=True)
                    if meta and meta.get("served_by"):
                        out.markdown(f"<div class='reply'>{html.escape(text)}</div>", unsafe_allow_html=True)
        else:
            with st.spinner("Waiting for the gateway…"):
                r = requests.post(f"{API}/v1/chat/completions", json=body, headers=hdr, timeout=180)
            data = r.json()
            meta = data.get("x_gateway")
            if r.status_code >= 400:
                out.error(data["error"]["message"])
            else:
                out.markdown(f"<div class='reply'>{html.escape(data['choices'][0]['message']['content'] or '')}</div>",
                             unsafe_allow_html=True)
        if meta:
            with meta_box.container():
                st.markdown(trace_html(meta), unsafe_allow_html=True)
                served = meta.get("served_by") or "nobody"
                bits = [f"served by **{served}**", f"{meta['attempts']} attempt(s)", ms(meta["latency_ms"]),
                        usd(meta["cost_usd"])]
                if meta.get("fallback"):
                    bits.insert(1, "**rescued by fallback**")
                if meta.get("cache_hit"):
                    bits.insert(1, "**cache hit**")
                st.caption(" · ".join(bits) + f" · `{meta['request_id']}`")

# ----------------------------------------------------------------------------- traffic
with tab_traffic:
    win = st.segmented_control("Window", ["15 min", "1 hour", "24 hours"], default="15 min", key="win")
    minutes = {"15 min": 15, "1 hour": 60, "24 hours": 1440}[win or "15 min"]

    @st.fragment(run_every=3)
    def traffic():
        s = api("GET", "/stats", params={"minutes": minutes})
        t = s["totals"]
        n = t["requests"] or 0
        k = st.columns(5)
        tile(k[0], "Requests", f"{n:,}", f"last {minutes} min")
        tile(k[1], "Success rate", f"{t['ok'] / n:.1%}" if n else "–", f"{t['errors']} failed")
        tile(k[2], "Rescued by fallback", f"{t['rescued']:,}", f"{t['rescued'] / n:.0%} of requests" if n else "")
        tile(k[3], "Latency p50 / p95", f"{ms(t['p50_ms'])} / {ms(t['p95_ms'])}",
             f"TTFT p50 {ms(t['ttft_p50_ms'])}")
        tile(k[4], "Cost", usd(t["cost_usd"]), f"{t['input_tokens']:,} in · {t['output_tokens']:,} out tokens")

        left, right = st.columns([1.5, 1])
        with left:
            st.subheader("Requests over time")
            tl = pd.DataFrame(s["timeline"])
            if tl.empty:
                st.info("No traffic yet: send something from the 💬 Playground or point a client at /v1.")
            else:
                tl["t"] = pd.to_datetime(tl["t"], utc=True)
                order = ["direct", "rescued", "cached", "failed"]
                labels = {"direct": "served by first choice", "rescued": "rescued by fallback", "cached": "cache hit",
                          "failed": "failed"}
                long = tl.melt("t", order, "outcome", "requests")
                long = long[long["requests"] > 0].copy()
                gap = pd.Timedelta(seconds=s["bucket_seconds"] * 0.12)
                long["t0"] = long["t"] + gap
                long["t1"] = long["t"] + pd.Timedelta(seconds=s["bucket_seconds"]) - gap
                long["rank"] = long["outcome"].map(order.index)
                long["outcome"] = long["outcome"].map(labels)
                domain = [pd.Timestamp(s["since"]).isoformat(), pd.Timestamp(s["now"]).isoformat()]
                bars = alt.Chart(long).mark_bar(stroke=SURFACE, strokeWidth=1).encode(
                    x=alt.X("t0:T", title=None, scale=alt.Scale(domain=domain),
                            axis=alt.Axis(format="%H:%M", labelColor=MUTED, grid=False, tickCount=6)),
                    x2="t1:T",
                    y=alt.Y("sum(requests):Q", title=f"requests per {s['bucket_seconds']} s", stack="zero",
                            axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor="#eee", tickMinStep=1)),
                    color=alt.Color("outcome:N", sort=[labels[o] for o in order],
                                    scale=alt.Scale(domain=[labels[o] for o in order],
                                                    range=[BLUE, ORANGE, AQUA, CRITICAL]),
                                    legend=alt.Legend(orient="top", title=None, labelColor=INK)),
                    order=alt.Order("rank:Q"),
                    tooltip=[alt.Tooltip("t:T", format="%H:%M:%S", title="from"), "outcome:N", "requests:Q"],
                )
                st.altair_chart(bars.properties(height=260).configure_view(strokeWidth=0), width="stretch")
        with right:
            st.subheader("Latency by deployment")
            deps = pd.DataFrame(s["deployments"])
            if not deps.empty and deps["p50_ms"].notna().any():
                dep_order = deps.sort_values("p50_ms")["deployment"].tolist()
                lat = deps.melt("deployment", ["p50_ms", "p95_ms"], "stat", "ms").dropna()
                lat["stat"] = lat["stat"].map({"p50_ms": "p50", "p95_ms": "p95"})
                rule = alt.Chart(deps.dropna(subset=["p50_ms"])).mark_rule(color="#d1d5db", strokeWidth=2).encode(
                    y=alt.Y("deployment:N", title=None, sort=dep_order), x="p50_ms:Q", x2="p95_ms:Q")
                dots = alt.Chart(lat).mark_circle(size=110, opacity=1, stroke=SURFACE, strokeWidth=2).encode(
                    y=alt.Y("deployment:N", title=None, sort=dep_order, axis=alt.Axis(labelColor=INK)),
                    x=alt.X("ms:Q", title="ms (successful attempts)",
                            axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor="#eee")),
                    color=alt.Color("stat:N", scale=alt.Scale(domain=["p50", "p95"], range=[BLUE, ORANGE]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["deployment", "stat", alt.Tooltip("ms:Q", format=",.0f")])
                st.altair_chart((rule + dots).properties(height=260).configure_view(strokeWidth=0), width="stretch")
            else:
                st.caption("No successful attempts in this window.")

        st.subheader("Deployments")
        if s["deployments"]:
            st.dataframe(pd.DataFrame([{
                "deployment": d["deployment"], "attempts": d["attempts"], "ok": d["ok"], "errors": d["errors"],
                "error kinds": ", ".join(f"{k} ×{v}" for k, v in d["errors_by_kind"].items()) or "–",
                "served (not cached)": d["served"], "p50": ms(d["p50_ms"]), "p95": ms(d["p95_ms"]), "cost": usd(d["cost_usd"]),
            } for d in s["deployments"]]), hide_index=True, width="stretch")

        st.subheader("Recent requests")
        reqs = api("GET", "/requests", params={"limit": 15})
        if reqs:
            st.dataframe(pd.DataFrame([{
                "time": r["created_at"][11:19], "route": r["route"],
                "outcome": ("⚡ cache" if r["cache_hit"] else "↪ fallback" if r["fallback"] else "✓ ok")
                if r["status"] == "ok" else f"✗ {r['error_kind']}",
                "served by": r["served_by"] or "–", "attempts": r["attempts"], "latency": ms(r["latency_ms"]),
                "tokens": f"{r['input_tokens']} → {r['output_tokens']}", "cost": usd(r["cost_usd"]),
                "prompt": r["prompt"], "id": r["id"]} for r in reqs]), hide_index=True, width="stretch")
    traffic()

    with st.expander("Inspect a request"):
        rid = st.text_input("Request id (gwr_…)", key="rid")
        if rid:
            d = api("GET", f"/requests/{rid.strip()}")
            if "api_error" in d:
                st.error(d["api_error"])
            else:
                st.markdown(trace_html({"skipped": d["skipped"], "trace": d["trace"]}), unsafe_allow_html=True)
                st.json({k: v for k, v in d.items() if k not in ("trace", "skipped")}, expanded=False)

# ----------------------------------------------------------------------------- deployments
with tab_deps:
    chaos = api("GET", "/chaos")
    if not chaos.get("enabled"):
        st.caption("Fault injection is off. Start the API with `ENABLE_CHAOS=true` to break deployments on purpose.")

    @st.fragment(run_every=2)
    def dep_cards():
        cols = st.columns(3)
        for i, d in enumerate(api("GET", "/deployments")):
            h = d["health"]
            state = {"closed": ("🟢", "circuit closed"), "half_open": ("🟡", "half-open: probing"),
                     "open": ("🔴", f"circuit open · retry in {h['reopens_in_s']:.0f}s" if h["reopens_in_s"] else "open")}
            icon, label = state[h["state"]]
            if h["rate_limited_for_s"]:
                label += f" · ⏳ rate limited {h['rate_limited_for_s']:.0f}s"
            strip = "".join(f"<span style='background:{GOOD if ok else CRITICAL}'></span>" for ok in h["recent"])                 or "<span style='background:#eee'></span>"
            fault = d["chaos"]
            fault_txt = (f"<div class='meta'>💥 injecting <b>{fault['kind']}</b> on {fault['rate']:.0%} of calls"
                         + (f", {fault['expires_in_s']:.0f}s left" if fault.get("expires_in_s") else "") + "</div>") if fault else ""
            err = (f"<div class='meta'>last error {h['last_error_ago_s']:.0f}s ago: {html.escape(h['last_error'][:90])}</div>"
                   if h["last_error"] else "")
            cols[i % 3].markdown(
                f"<div class='dep'><div class='name'>{icon} {d['name']}</div>"
                f"<div class='meta'>{d['provider']} · {d['model']} · ${d['price']['input']:g}/${d['price']['output']:g} per Mtok</div>"
                f"<div style='margin:.4rem 0'>{label} · avg {ms(h['latency_ms'])} · {h['successes']} ok / {h['errors']} errors</div>"
                f"<div class='strip'>{strip}</div>{fault_txt}{err}</div>",
                unsafe_allow_html=True)
    dep_cards()

    if chaos.get("enabled"):
        st.subheader("Break something")
        with st.form("chaos"):
            c = st.columns([1.2, 1.2, 1, 1, 1])
            dep = c[0].selectbox("Deployment", [d["name"] for d in deployments])
            kind = c[1].selectbox("Fault", ["overloaded", "rate_limited", "server_error", "timeout", "auth", "refusal",
                                            "midstream", "latency"])
            rate = c[2].slider("Share of calls", 0.0, 1.0, 1.0, 0.1)
            dur = c[3].number_input("For (s)", 5, 3600, 60)
            lat = c[4].number_input("Extra latency (ms)", 0, 30000, 0, 100)
            if st.form_submit_button("Inject", type="primary"):
                res = api("POST", "/chaos", json={"deployment": dep, "kind": kind, "rate": rate, "duration_s": dur,
                                                  "latency_ms": lat})
                st.error(res["api_error"]) if "api_error" in res else st.rerun()
        b = st.columns(4)
        if b[0].button("Clear all faults"):
            api("DELETE", "/chaos")
            st.rerun()
        reset = b[1].selectbox("Reset circuit", ["–"] + [d["name"] for d in deployments], label_visibility="collapsed")
        if reset != "–" and b[2].button(f"Reset {reset}"):
            api("POST", f"/deployments/{reset}/reset")
            st.rerun()

# ----------------------------------------------------------------------------- keys
with tab_keys:
    keys = api("GET", "/keys")
    if isinstance(keys, dict) and "api_error" in keys:
        st.warning(f"{keys['api_error']} — set GATEWAY_ADMIN_KEY for the UI too.")
    else:
        if keys:
            st.dataframe(pd.DataFrame([{
                "name": k["name"], "key": k["prefix"] + "…", "budget / month": usd(k["monthly_budget_usd"])
                if k["monthly_budget_usd"] else "none",
                "spent this month": usd(k["spent_this_month_usd"]), "requests": k["requests_this_month"],
                "rpm": k["rpm"] or "–", "routes": ", ".join(k["routes"] or ["all"]),
                "status": "disabled" if k["disabled"] else "active", "id": k["id"]} for k in keys]),
                hide_index=True, width="stretch")
        else:
            st.info("No keys yet. Without REQUIRE_KEYS=true, anonymous requests are allowed (local development).")
        with st.form("newkey"):
            c = st.columns([2, 1, 1, 2])
            name = c[0].text_input("Name", placeholder="team-search")
            budget = c[1].number_input("Monthly budget ($, 0 = none)", 0.0, 100000.0, 25.0)
            rpm = c[2].number_input("Requests/min (0 = none)", 0, 100000, 120)
            allowed = c[3].multiselect("Routes (empty = all)", [r["name"] for r in routes])
            if st.form_submit_button("Create key", type="primary") and name:
                res = api("POST", "/keys", json={"name": name, "monthly_budget_usd": budget or None, "rpm": rpm or None,
                                                 "routes": allowed or None})
                if "api_error" in res:
                    st.error(res["api_error"])
                else:
                    st.success("Copy this key now: the gateway stores only its hash.")
                    st.code(res["key"], language=None)
        if keys:
            c = st.columns([2, 1, 3])
            pick = c[0].selectbox("Key", [k["id"] for k in keys],
                                  format_func=lambda i: next(k["name"] for k in keys if k["id"] == i))
            cur = next(k for k in keys if k["id"] == pick)
            if c[1].button("Enable" if cur["disabled"] else "Disable"):
                api("PATCH", f"/keys/{pick}", json={"disabled": not cur["disabled"]})
                st.rerun()
