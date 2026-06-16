from __future__ import annotations

import json
import os
import re

import httpx

# The coach's LLM, with failover like phalanx: a primary provider (Groq) and a fallback (Gemini).
# Both are OpenAI-compatible chat endpoints. If the primary errors or rate-limits (429), the call
# falls through to the secondary; if both fail it returns "" and the coach uses its heuristic. The
# whole thing is off unless a primary key is configured, so the app stays deterministic until wired.

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# primary (Groq by default)
_KEY = os.environ.get("PITCH_LLM_KEY", "")
_URL = os.environ.get("PITCH_LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
_MODEL = os.environ.get("PITCH_MODEL", "llama-3.3-70b-versatile")
# fallback (Gemini by default)
_KEY2 = os.environ.get("PITCH_LLM2_KEY", "")
_URL2 = os.environ.get("PITCH_LLM2_URL", _GEMINI_URL)
_MODEL2 = os.environ.get("PITCH_MODEL2", "gemini-2.0-flash")


def enabled() -> bool:
    return bool(_KEY or _KEY2)


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
    if r.status_code >= 300:  # raise so the caller falls through to the fallback provider
        raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
    return r.json()["choices"][0]["message"]["content"]


async def complete(system: str, user: str, timeout: float = 8.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as h:
        for key, url, model in ((_KEY, _URL, _MODEL), (_KEY2, _URL2, _MODEL2)):
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
