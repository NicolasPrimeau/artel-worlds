from __future__ import annotations

import datetime
import json
import os
import re

import httpx

# The coach's LLM, with failover like phalanx: a primary provider (Groq) and a fallback (Gemini).
# Both are OpenAI-compatible chat endpoints. If the primary errors or rate-limits (429), the call
# falls through to the secondary; if both fail it returns "" and the coach uses its heuristic. The
# whole thing is off unless a primary key is configured, so the app stays deterministic until wired.
# Every call's token usage is costed and ledgered (so the ops dashboard can show pitch's LLM spend),
# and a spend cap turns the LLM off once it's hit — the coach just falls back to its heuristic.

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# primary (Groq by default)
_KEY = os.environ.get("PITCH_LLM_KEY", "")
_URL = os.environ.get("PITCH_LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.environ.get("PITCH_MODEL", "llama-3.3-70b-versatile")
# fallback (Gemini by default)
_KEY2 = os.environ.get("PITCH_LLM2_KEY", "")
_URL2 = os.environ.get("PITCH_LLM2_URL", _GEMINI_URL)
MODEL2 = os.environ.get("PITCH_MODEL2", "gemini-2.0-flash")

# token pricing (USD per million) and the since-boot spend cap
_COST_IN = float(os.environ.get("PITCH_COST_IN_PER_M", "0.15")) / 1e6
_COST_OUT = float(os.environ.get("PITCH_COST_OUT_PER_M", "0.75")) / 1e6
CAP = float(os.environ.get("PITCH_SPEND_CAP_USD", "8"))

# running ledger, exposed via the world's /debug for the ops dashboard
SPEND: dict = {"usd": 0.0, "in": 0, "out": 0, "calls": 0, "throttled": 0, "days": {}}


def within_cap() -> bool:
    return SPEND["usd"] < CAP


def enabled() -> bool:
    return bool(_KEY or _KEY2) and within_cap()


def _account(usage: dict) -> None:
    pin = int(usage.get("prompt_tokens", 0) or 0)
    pout = int(usage.get("completion_tokens", 0) or 0)
    cost = pin * _COST_IN + pout * _COST_OUT
    SPEND["usd"] = round(SPEND["usd"] + cost, 6)
    SPEND["in"] += pin
    SPEND["out"] += pout
    SPEND["calls"] += 1
    day = datetime.date.today().isoformat()
    SPEND["days"][day] = round(SPEND["days"].get(day, 0.0) + cost, 6)
    for old in sorted(SPEND["days"])[:-30]:  # keep ~30 days
        del SPEND["days"][old]


async def _ask(h: httpx.AsyncClient, key: str, url: str, model: str, system: str, user: str) -> str:
    if not key:
        return ""
    r = await h.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": 0.4,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    if r.status_code == 429:
        SPEND["throttled"] += 1
    if r.status_code >= 300:  # raise so the caller falls through to the fallback provider
        raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
    data = r.json()
    _account(data.get("usage") or {})
    return data["choices"][0]["message"]["content"]


async def complete(system: str, user: str, timeout: float = 8.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as h:
        for key, url, model in ((_KEY, _URL, MODEL), (_KEY2, _URL2, MODEL2)):
            try:
                out = await _ask(h, key, url, model, system, user)
                if out:
                    return out
            except Exception:
                continue
    return ""


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except Exception:
        return None
