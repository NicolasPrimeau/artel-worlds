from __future__ import annotations

import json
import os
import re

import httpx

# A tiny LLM client for the coach. OpenAI-compatible (e.g. Gemini Flash via PITCH_LLM_KEY/URL) or
# the Anthropic API. Disabled unless a key is set, so the app stays deterministic until configured.
# Every call is best-effort: on any error it returns "" and the coach falls back to its heuristic.

PROVIDER = os.environ.get("PITCH_LLM_PROVIDER", "openai")
MODEL = os.environ.get("PITCH_MODEL", "gemini-2.0-flash")
_KEY = os.environ.get("PITCH_LLM_KEY", "")
_URL = os.environ.get(
    "PITCH_LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
_ANTHROPIC = os.environ.get("PITCH_ANTHROPIC_KEY", "")  # deliberately NOT ANTHROPIC_API_KEY


def enabled() -> bool:
    return bool(_KEY or _ANTHROPIC)


async def complete(system: str, user: str, timeout: float = 8.0) -> str:
    try:
        async with httpx.AsyncClient(timeout=timeout) as h:
            if _KEY:
                r = await h.post(
                    _URL,
                    headers={"Authorization": f"Bearer {_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.5,
                    },
                )
                return r.json()["choices"][0]["message"]["content"] if r.status_code < 300 else ""
            if _ANTHROPIC:
                r = await h.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": _ANTHROPIC,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": MODEL,
                        "max_tokens": 300,
                        "system": system,
                        "messages": [{"role": "user", "content": user}],
                    },
                )
                return r.json()["content"][0]["text"] if r.status_code < 300 else ""
    except Exception:
        return ""
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
