from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from dataclasses import dataclass

import httpx

from .models import Model

# A generic LLM router shared across worlds. Agents are decoupled from any specific model: they hand a
# standardized Request to the router, which round-robins it across a pool of OpenAI-compatible models,
# skips any model still cooling down from a 429, and honours per-request requirements (tool support,
# free-vs-paid). It records per-model telemetry so the ops board can show exactly what's throttled.


GRADE_RANK = {"fast": 0, "balanced": 1, "capable": 2}


@dataclass
class Request:
    system: str
    user: str
    temperature: float = 0.7
    timeout: float = 16.0
    requires_tools: bool = False  # if set, only models whose caps mark tool support are eligible
    allow_paid: bool = False  # if unset (default), paid-tier models are never routed to
    min_grade: str = (
        "fast"  # "fast" | "balanced" | "capable" — filters to models at or above this tier
    )
    tools: list | None = None  # OpenAI tool schemas; when set, act() returns the model's tool call


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after


class Router:
    def __init__(
        self,
        models: list[Model],
        *,
        concurrency: int = 8,
        cooldown: float = 8.0,
        max_cooldown: float = 30.0,
        max_wait: float = 3.0,
        window: int = 5,
        cost_in_per_m: float = 0.15,
        cost_out_per_m: float = 0.60,
        shaper=None,
    ):
        self.models = list(models)
        self._cooldown = cooldown
        self._max_cooldown = max_cooldown
        self._max_wait = max_wait
        self._window = window
        self._cost_in = cost_in_per_m / 1e6
        self._cost_out = cost_out_per_m / 1e6
        self._shaper = shaper  # optional fn(user, model_id) -> user, for per-model request shaping
        self._sem = asyncio.Semaphore(concurrency)
        self._lock = asyncio.Lock()
        self._cursor = 0
        self.spend = {"usd": 0.0, "in": 0, "out": 0, "calls": 0, "days": {}}

    def enabled(self) -> bool:
        return bool(self.models)

    def describe(self) -> str:
        if not self.models:
            return "no LLM configured"
        provs = dict.fromkeys(m.provider for m in self.models)
        return f"{len(self.models)}-model round-robin mesh: " + ", ".join(
            f"{p}×{sum(1 for m in self.models if m.provider == p)}" for p in provs
        )

    def metrics(self) -> list[dict]:
        # per-model view for the ops board: counters, the rolling outcome window, and live throttle state
        now = time.monotonic()
        out = []
        for m in self.models:
            cooling = m.cooldown > now
            out.append(
                {
                    "provider": m.provider,
                    "model": m.model,
                    "tier": m.tier,
                    "grade": m.grade,
                    "tools": m.tools,
                    "calls": m.calls,
                    "ok": m.ok,
                    "throttled": m.throttled,
                    "errors": m.errors,
                    "recent": list(m.recent),
                    "cooling": cooling,
                    "cools_in": round(m.cooldown - now, 1) if cooling else 0.0,
                }
            )
        return out

    def _eligible(self, req: Request) -> list[Model]:
        min_rank = GRADE_RANK.get(req.min_grade, 0)
        return [
            m
            for m in self.models
            if (not req.requires_tools or m.tools)
            and (req.allow_paid or m.tier == "free")
            and GRADE_RANK.get(m.grade, 0) >= min_rank
        ]

    async def _pick(self, eligible: list[Model]) -> Model | None:
        # round-robin over the eligible subset, skipping models still cooling from a 429. If every
        # eligible model is cooling, return the one that frees up soonest rather than failing outright.
        if not eligible:
            return None
        ids = {id(m) for m in eligible}
        async with self._lock:
            now = time.monotonic()
            n = len(self.models)
            soonest: Model | None = None
            for _ in range(n):
                m = self.models[self._cursor % n]
                self._cursor = (self._cursor + 1) % n
                if id(m) not in ids:
                    continue
                if m.cooldown <= now:
                    return m
                if soonest is None or m.cooldown < soonest.cooldown:
                    soonest = m
            return soonest

    def _record(self, m: Model, status: str) -> None:
        m.calls += 1
        if status == "ok":
            m.ok += 1
        elif status == "429":
            m.throttled += 1
        else:
            m.errors += 1
        m.recent.append(status)
        del m.recent[: -self._window]

    def _account(self, usage: dict) -> None:
        pin = int(usage.get("prompt_tokens", 0) or 0)
        pout = int(usage.get("completion_tokens", 0) or 0)
        cost = pin * self._cost_in + pout * self._cost_out
        self.spend["usd"] = round(self.spend["usd"] + cost, 6)
        self.spend["in"] += pin
        self.spend["out"] += pout
        self.spend["calls"] += 1
        day = datetime.date.today().isoformat()
        self.spend["days"][day] = round(self.spend["days"].get(day, 0.0) + cost, 6)
        for old in sorted(self.spend["days"])[:-30]:
            del self.spend["days"][old]

    async def _call(self, h: httpx.AsyncClient, m: Model, req: Request) -> dict:
        # one POST; returns the assistant message (content and/or tool_calls). Raises on 429 / bad status.
        user = self._shaper(req.user, m.model) if self._shaper else req.user
        payload = {
            "model": m.model,
            "temperature": req.temperature,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": user},
            ],
        }
        if req.tools:
            payload["tools"] = req.tools
            payload["tool_choice"] = "auto"
        r = await h.post(m.url, headers={"Authorization": f"Bearer {m.key}"}, json=payload)
        if r.status_code == 429:
            raise RateLimited(float(r.headers.get("retry-after", "0") or 0))
        if r.status_code >= 300:
            raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
        data = r.json()
        self._account(data.get("usage") or {})
        return data["choices"][0]["message"]

    async def _run(self, req: Request, extract):
        # shared round-robin/cooldown loop; `extract(msg)` returns the result or None (a miss → next model).
        # returns (result, model_id) on success, (None, None) on miss — callers that don't want the model
        # just unwrap the first element.
        eligible = self._eligible(req)
        if not eligible:
            return None, None
        tries = max(4, len(eligible) + 1)
        async with self._sem, httpx.AsyncClient(timeout=req.timeout) as h:
            for _ in range(tries):
                m = await self._pick(eligible)
                if m is None:
                    return None, None
                wait = m.cooldown - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(min(wait, self._max_wait))
                try:
                    got = extract(await self._call(h, m, req))
                    if got is not None:
                        self._record(m, "ok")
                        return got, m.model
                    self._record(m, "err")
                except RateLimited as ex:
                    m.cooldown = time.monotonic() + min(
                        max(ex.retry_after, self._cooldown), self._max_cooldown
                    )
                    self._record(m, "429")
                except Exception:
                    self._record(m, "err")
        return None, None

    @staticmethod
    def _text(msg):
        out = (msg.get("content") or "").strip()
        return out or None

    @staticmethod
    def _tool(msg):
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            if not name:
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            return {"name": name, "args": args if isinstance(args, dict) else {}}
        return None

    async def complete(self, req: Request) -> str:
        res, _ = await self._run(req, self._text)
        return res or ""

    async def complete_m(self, req: Request) -> tuple:
        # like complete(), but returns (text, model_id) so callers can show which model spoke
        res, model = await self._run(req, self._text)
        return (res or ""), model

    async def act_m(self, req: Request) -> tuple:
        return await self._run(req, self._tool)  # (action|None, model_id|None)

    async def complete_many_m(self, reqs: list[Request]) -> list[tuple]:
        return await asyncio.gather(*(self.complete_m(r) for r in reqs))

    async def act_many_m(self, reqs: list[Request]) -> list[tuple]:
        return await asyncio.gather(*(self.act_m(r) for r in reqs))

    async def act(self, req: Request) -> dict | None:
        # returns the first tool call as {"name": str, "args": dict}, or None if no model produced one.
        res, _ = await self._run(req, self._tool)
        return res

    async def complete_many(self, reqs: list[Request]) -> list[str]:
        return await asyncio.gather(*(self.complete(r) for r in reqs))

    async def act_many(self, reqs: list[Request]) -> list[dict | None]:
        return await asyncio.gather(*(self.act(r) for r in reqs))


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except Exception:
        return None
