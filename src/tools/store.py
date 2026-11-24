"""PostgreSQL: virtual API keys, the request log (one row per request, one per attempt) and the response cache."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

metadata = MetaData()

api_keys = Table(
    "api_keys", metadata,
    Column("id", String(12), primary_key=True),
    Column("name", Text, nullable=False),
    Column("key_hash", String(64), nullable=False, unique=True),   # sha256 of the key; the key itself is never stored
    Column("prefix", String(12), nullable=False),                  # "gw-1a2b3c" to recognise it in the UI
    Column("monthly_budget_usd", Float),
    Column("rpm", Integer),
    Column("routes", JSONB),                                       # allowed routes; null = all
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
requests_t = Table(
    "requests", metadata,
    Column("id", String(24), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("key_id", String(12), ForeignKey("api_keys.id", ondelete="SET NULL")),
    Column("route", Text, nullable=False),
    Column("served_by", Text),                 # deployment that answered
    Column("model", Text),                     # provider model id that answered
    Column("status", String(8), nullable=False),   # ok | error
    Column("http_status", Integer, nullable=False),
    Column("error_kind", Text),
    Column("error", Text),
    Column("stream", Boolean, nullable=False, default=False),
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("fallback", Boolean, nullable=False, default=False),    # answered by a deployment other than the first choice
    Column("attempts", Integer, nullable=False, default=0),
    Column("latency_ms", Float),
    Column("ttft_ms", Float),
    Column("input_tokens", Integer, default=0),
    Column("output_tokens", Integer, default=0),
    Column("cost_usd", Float, default=0.0),
    Column("finish_reason", Text),
    Column("prompt", Text),
    Column("reply", Text),
    Column("skipped", JSONB, nullable=False, default=list),       # deployments the planner passed over, and why
    Index("ix_requests_created", "created_at"),
    Index("ix_requests_key_created", "key_id", "created_at"),
)
attempts_t = Table(
    "attempts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("request_id", String(24), ForeignKey("requests.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("n", Integer, nullable=False),
    Column("deployment", Text, nullable=False),
    Column("outcome", String(8), nullable=False),     # ok | error
    Column("error_kind", Text),
    Column("status_code", Integer),
    Column("detail", Text),
    Column("injected", Boolean, nullable=False, default=False),
    Column("latency_ms", Float),
    Column("started_at", DateTime(timezone=True), nullable=False),
)
cache_t = Table(
    "cache", metadata,
    Column("key", String(64), primary_key=True),
    Column("route", Text, nullable=False),
    Column("response", JSONB, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Column("hits", Integer, nullable=False, default=0),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _row(r) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in r._mapping.items()}


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def month_start(t: datetime | None = None) -> datetime:
    t = t or now()
    return t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class Store:
    def __init__(self, database_url: str):
        url = database_url
        for p in ("postgresql://", "postgres://"):
            if url.startswith(p):
                url = "postgresql+psycopg://" + url[len(p):]
        self.engine: Engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)

    def init(self) -> None:
        metadata.create_all(self.engine)

    # ------------------------------------------------------------------ keys
    def create_key(self, name: str, monthly_budget_usd: float | None = None, rpm: int | None = None,
                   routes: list[str] | None = None) -> tuple[dict, str]:
        key = "gw-" + secrets.token_urlsafe(24)
        row = {"id": uuid.uuid4().hex[:12], "name": name, "key_hash": hash_key(key), "prefix": key[:9],
               "monthly_budget_usd": monthly_budget_usd, "rpm": rpm, "routes": routes, "disabled": False,
               "created_at": now()}
        with self.engine.begin() as c:
            c.execute(insert(api_keys).values(**row))
        return self.key(row["id"]), key

    def key(self, kid: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(api_keys).where(api_keys.c.id == kid)).first()
        return _row(r) if r else None

    def key_by_secret(self, secret: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(api_keys).where(api_keys.c.key_hash == hash_key(secret))).first()
        return _row(r) if r else None

    def keys(self) -> list[dict]:
        start = month_start()
        spend = (select(requests_t.c.key_id, func.coalesce(func.sum(requests_t.c.cost_usd), 0).label("spent"),
                        func.count().label("requests"))
                 .where(requests_t.c.created_at >= start).group_by(requests_t.c.key_id).subquery())
        q = (select(api_keys, func.coalesce(spend.c.spent, 0).label("spent_this_month_usd"),
                    func.coalesce(spend.c.requests, 0).label("requests_this_month"))
             .select_from(api_keys.outerjoin(spend, spend.c.key_id == api_keys.c.id))
             .order_by(api_keys.c.created_at))
        with self.engine.connect() as c:
            return [{k: v for k, v in _row(r).items() if k != "key_hash"} for r in c.execute(q)]

    def update_key(self, kid: str, **values) -> dict | None:
        with self.engine.begin() as c:
            if not c.execute(update(api_keys).where(api_keys.c.id == kid).values(**values)).rowcount:
                return None
        return self.key(kid)

    def delete_key(self, kid: str) -> bool:
        with self.engine.begin() as c:
            return bool(c.execute(delete(api_keys).where(api_keys.c.id == kid)).rowcount)

    def spent_this_month(self, kid: str) -> float:
        with self.engine.connect() as c:
            return float(c.execute(select(func.coalesce(func.sum(requests_t.c.cost_usd), 0))
                                   .where(requests_t.c.key_id == kid, requests_t.c.created_at >= month_start()))
                         .scalar())

    # ------------------------------------------------------------------ request log
    def log_request(self, row: dict, attempts: list[dict]) -> None:
        with self.engine.begin() as c:
            c.execute(insert(requests_t).values(**row))
            if attempts:
                c.execute(insert(attempts_t), [{**a, "request_id": row["id"]} for a in attempts])

    def requests(self, limit: int = 100, route: str | None = None, status: str | None = None,
                 fallback: bool | None = None, key_id: str | None = None) -> list[dict]:
        q = select(requests_t).order_by(requests_t.c.created_at.desc()).limit(limit)
        if route:
            q = q.where(requests_t.c.route == route)
        if status:
            q = q.where(requests_t.c.status == status)
        if fallback is not None:
            q = q.where(requests_t.c.fallback == fallback)
        if key_id:
            q = q.where(requests_t.c.key_id == key_id)
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(q)]

    def request(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(requests_t).where(requests_t.c.id == rid)).first()
            if not r:
                return None
            out = _row(r)
            out["trace"] = [_row(a) for a in c.execute(select(attempts_t).where(attempts_t.c.request_id == rid)
                                                       .order_by(attempts_t.c.n))]
        return out

    # ------------------------------------------------------------------ cache
    def cache_get(self, key: str) -> dict | None:
        with self.engine.begin() as c:
            r = c.execute(update(cache_t).where(cache_t.c.key == key, cache_t.c.expires_at > now())
                          .values(hits=cache_t.c.hits + 1).returning(cache_t.c.response)).first()
        return r[0] if r else None

    def cache_put(self, key: str, route: str, response: dict, ttl_s: int) -> None:
        t = now()
        values = {"route": route, "response": response, "created_at": t, "expires_at": t + timedelta(seconds=ttl_s),
                  "hits": 0}
        with self.engine.begin() as c:
            c.execute(pg_insert(cache_t).values(key=key, **values)
                      .on_conflict_do_update(index_elements=["key"], set_=values))
            c.execute(delete(cache_t).where(cache_t.c.expires_at < t))

    def cache_clear(self) -> int:
        with self.engine.begin() as c:
            return c.execute(delete(cache_t)).rowcount

    # ------------------------------------------------------------------ stats
    def stats(self, minutes: int = 60) -> dict:
        since = now() - timedelta(minutes=minutes)
        p = {"since": since}
        with self.engine.connect() as c:
            totals = c.execute(text("""
                SELECT count(*) AS requests,
                       count(*) FILTER (WHERE status = 'ok') AS ok,
                       count(*) FILTER (WHERE status = 'error') AS errors,
                       count(*) FILTER (WHERE fallback AND status = 'ok') AS rescued,
                       count(*) FILTER (WHERE cache_hit) AS cache_hits,
                       coalesce(sum(cost_usd), 0) AS cost_usd,
                       coalesce(sum(input_tokens), 0) AS input_tokens,
                       coalesce(sum(output_tokens), 0) AS output_tokens,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit) AS p50_ms,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE NOT cache_hit) AS p95_ms,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY ttft_ms) FILTER (WHERE stream AND NOT cache_hit)
                           AS ttft_p50_ms
                FROM requests WHERE created_at >= :since"""), p).mappings().one()
            routes = c.execute(text("""
                SELECT route, count(*) AS requests,
                       count(*) FILTER (WHERE status = 'error') AS errors,
                       count(*) FILTER (WHERE fallback AND status = 'ok') AS rescued,
                       count(*) FILTER (WHERE cache_hit) AS cache_hits,
                       coalesce(sum(cost_usd), 0) AS cost_usd,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
                FROM requests WHERE created_at >= :since GROUP BY route ORDER BY count(*) DESC"""), p).mappings().all()
            deps = c.execute(text("""
                SELECT a.deployment, count(*) AS attempts,
                       count(*) FILTER (WHERE a.outcome = 'ok') AS ok,
                       count(*) FILTER (WHERE a.outcome = 'error') AS errors,
                       percentile_cont(0.5) WITHIN GROUP (ORDER BY a.latency_ms) FILTER (WHERE a.outcome = 'ok') AS p50_ms,
                       percentile_cont(0.95) WITHIN GROUP (ORDER BY a.latency_ms) FILTER (WHERE a.outcome = 'ok') AS p95_ms
                FROM attempts a WHERE a.started_at >= :since GROUP BY a.deployment ORDER BY a.deployment"""),
                p).mappings().all()
            kinds = c.execute(text("""
                SELECT deployment, error_kind, count(*) AS n FROM attempts
                WHERE started_at >= :since AND error_kind IS NOT NULL GROUP BY deployment, error_kind"""), p).mappings().all()
            served = c.execute(text("""
                SELECT served_by, count(*) AS n, coalesce(sum(cost_usd), 0) AS cost_usd FROM requests
                WHERE created_at >= :since AND status = 'ok' AND NOT cache_hit AND served_by IS NOT NULL
                GROUP BY served_by"""),
                p).mappings().all()
            bucket = 900 if minutes > 360 else (60 if minutes > 30 else 15)
            timeline = c.execute(text("""
                SELECT to_timestamp(floor(extract(epoch FROM created_at) / :b) * :b) AS t,
                       count(*) FILTER (WHERE status = 'ok' AND NOT fallback AND NOT cache_hit) AS direct,
                       count(*) FILTER (WHERE status = 'ok' AND fallback) AS rescued,
                       count(*) FILTER (WHERE cache_hit) AS cached,
                       count(*) FILTER (WHERE status = 'error') AS failed
                FROM requests WHERE created_at >= :since GROUP BY 1 ORDER BY 1"""), {**p, "b": bucket}).mappings().all()
        by_kind: dict[str, dict] = {}
        for k in kinds:
            by_kind.setdefault(k["deployment"], {})[k["error_kind"]] = k["n"]
        served_map = {s["served_by"]: s for s in served}

        def clean(d):
            return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in dict(d).items()}

        return {
            "window_minutes": minutes, "bucket_seconds": bucket, "since": since.isoformat(), "now": now().isoformat(),
            "totals": clean(totals),
            "routes": [clean(r) for r in routes],
            "deployments": [clean({**dict(d),
                                   "errors_by_kind": by_kind.get(d["deployment"], {}),
                                   "served": served_map.get(d["deployment"], {}).get("n", 0),
                                   "cost_usd": served_map.get(d["deployment"], {}).get("cost_usd", 0.0)})
                            for d in deps],
            "timeline": [{**dict(t), "t": t["t"].isoformat()} for t in timeline],
        }

    def clear_log(self) -> None:
        with self.engine.begin() as c:
            c.execute(text("TRUNCATE requests, attempts RESTART IDENTITY CASCADE"))
