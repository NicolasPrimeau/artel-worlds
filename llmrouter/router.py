from __future__ import annotations

import asyncio
import datetime
import json
import os
import re
import time
from dataclasses import dataclass

import httpx

# Every world's LLM calls go through OpenRouter. It owns provider choice and fallback between the
# models in a request; this client adds the world-side concerns: free-only enforcement, a pause
# after a 429 so a spent quota isn't hammered, and the per-model telemetry the ops board shows.

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-31b-it:free",
)


@dataclass
class Request:
    system: str
    user: str
    temperature: float = 0.7
    timeout: float = 16.0
    tools: list | None = None  # OpenAI tool schemas; when set, act() returns the model's tool call
    tool_choice: str | None = None  # "auto" (default) | "required" | "none"
    reasoning_effort: str | None = None  # "none" turns reasoning off where the model allows it
    max_tokens: int | None = None
    deadline: float | None = None  # overall wall-clock budget (s) across attempts


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after


def pool(spec: str | None) -> list[str]:
    return [m.strip() for m in (spec or "").split(",") if m.strip()] or list(DEFAULT_MODELS)


def _stat() -> dict:
    return {"calls": 0, "ok": 0, "throttled": 0, "errors": 0, "recent": []}


class Router:
    def __init__(
        self,
        models: list[str] | None = None,
        *,
        key: str | None = None,
        concurrency: int = 8,
        cooldown: float = 8.0,
        max_cooldown: float = 60.0,
        window: int = 5,
        reasoning: str | None = None,
    ):
        self.models = list(models or DEFAULT_MODELS)
        paid = [m for m in self.models if not m.endswith(":free")]
        if paid:
            raise ValueError(f"paid models are not allowed: {paid}")
        self.key = os.environ.get("OPENROUTER_API_KEY", "") if key is None else key
        self._cooldown = cooldown
        self._max_cooldown = max_cooldown
        self._window = window
        self._reasoning = reasoning
        self._sem = asyncio.Semaphore(concurrency)
        self._paused_until = 0.0
        self._stats = {m: _stat() for m in self.models}
        self.spend = {"usd": 0.0, "in": 0, "out": 0, "calls": 0, "days": {}}

    def enabled(self) -> bool:
        return bool(self.key)

    def describe(self) -> str:
        if not self.enabled():
            return "no LLM configured"
        return f"OpenRouter: {len(self.models)} free models"

    def metrics(self) -> list[dict]:
        wait = self._paused_until - time.monotonic()
        return [
            {
                "provider": "openrouter",
                "model": model,
                "tier": "free",
                "tools": True,
                **{k: (list(v) if k == "recent" else v) for k, v in stat.items()},
                "cooling": wait > 0,
                "cools_in": round(wait, 1) if wait > 0 else 0.0,
            }
            for model, stat in self._stats.items()
        ]

    def _record(self, model: str, status: str) -> None:
        stat = self._stats.setdefault(model, _stat())
        stat["calls"] += 1
        key = {"ok": "ok", "429": "throttled"}.get(status, "errors")
        stat[key] += 1
        stat["recent"].append(status)
        del stat["recent"][: -self._window]

    def _account(self, usage: dict) -> None:
        cost = float(usage.get("cost", 0) or 0)
        self.spend["usd"] += cost
        self.spend["in"] += int(usage.get("prompt_tokens", 0) or 0)
        self.spend["out"] += int(usage.get("completion_tokens", 0) or 0)
        self.spend["calls"] += 1
        day = datetime.date.today().isoformat()
        self.spend["days"][day] = self.spend["days"].get(day, 0.0) + cost
        for old in sorted(self.spend["days"])[:-30]:
            del self.spend["days"][old]

    def _payload(self, req: Request) -> dict:
        payload = {
            "model": self.models[0],
            "models": self.models,
            "temperature": req.temperature,
            "usage": {"include": True},
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
        }
        effort = req.reasoning_effort or self._reasoning
        if effort == "none":
            payload["reasoning"] = {"enabled": False}
        elif effort:
            payload["reasoning"] = {"effort": effort, "exclude": True}
        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens
        if req.tools:
            payload["tools"] = req.tools
            payload["tool_choice"] = req.tool_choice or "auto"
            payload["provider"] = {"require_parameters": True}
        return payload

    async def _call(self, h: httpx.AsyncClient, req: Request) -> tuple[dict, str]:
        r = await h.post(
            URL, headers={"Authorization": f"Bearer {self.key}"}, json=self._payload(req)
        )
        if r.status_code == 429:
            raise RateLimited(float(r.headers.get("retry-after", "0") or 0))
        if r.status_code >= 300:
            raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
        data = r.json()
        self._account(data.get("usage") or {})
        return data["choices"][0]["message"], data.get("model") or self.models[0]

    async def _run(self, req: Request, extract):
        if not self.enabled() or time.monotonic() < self._paused_until:
            return None, None
        start = time.monotonic()
        async with self._sem, httpx.AsyncClient(timeout=req.timeout) as h:
            for _ in range(2):
                if req.deadline is not None and time.monotonic() - start > req.deadline:
                    break
                model = self.models[0]
                try:
                    message, model = await self._call(h, req)
                    got = extract(message)
                    self._record(model, "ok" if got is not None else "err")
                    if got is not None:
                        return got, model
                except RateLimited as ex:
                    wait = min(max(ex.retry_after, self._cooldown), self._max_cooldown)
                    self._paused_until = time.monotonic() + wait
                    self._record(model, "429")
                    break
                except (httpx.HTTPError, KeyError, IndexError, ValueError):
                    self._record(model, "err")
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
            except ValueError:
                args = {}
            return {"name": name, "args": args if isinstance(args, dict) else {}}
        return None

    async def complete(self, req: Request) -> str:
        res, _ = await self._run(req, self._text)
        return res or ""

    async def complete_m(self, req: Request) -> tuple:
        res, model = await self._run(req, self._text)
        return (res or ""), model

    async def act_m(self, req: Request) -> tuple:
        return await self._run(req, self._tool)

    async def complete_many_m(self, reqs: list[Request]) -> list[tuple]:
        return await asyncio.gather(*(self.complete_m(r) for r in reqs))

    async def act_many_m(self, reqs: list[Request]) -> list[tuple]:
        return await asyncio.gather(*(self.act_m(r) for r in reqs))

    async def act(self, req: Request) -> dict | None:
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
    except ValueError:
        return None
