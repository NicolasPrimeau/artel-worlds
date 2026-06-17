from __future__ import annotations

import asyncio
import datetime
import json
import os
import re

import httpx

# The meeting's LLM. Same shape as pitch: an OpenAI-compatible chat endpoint (Groq by default) with an
# optional fallback. Off unless a key is configured — with no key the world falls back to the
# deterministic decider. Many agents speak per meeting, so calls are issued concurrently.

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

_KEY = os.environ.get("ALIBI_LLM_KEY", "")
_URL = os.environ.get("ALIBI_LLM_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.environ.get("ALIBI_MODEL", "openai/gpt-oss-120b")
_KEY2 = os.environ.get("ALIBI_LLM2_KEY", "")
_URL2 = os.environ.get("ALIBI_LLM2_URL", _GEMINI_URL)
MODEL2 = os.environ.get("ALIBI_MODEL2", "gemini-2.0-flash")

# Many agents speak per meeting; firing them all at once trips the provider's rate limit and calls come
# back empty. Cap in-flight calls and retry 429s with backoff so a meeting never silently loses votes.
_CONCURRENCY = int(os.environ.get("ALIBI_CONCURRENCY", "8"))
_RETRIES = 4
_sem: asyncio.Semaphore | None = None

# token + cost ledger for the ops cost metric. Groq's gpt-oss is cheap (often free tier), so this is a
# small estimate using a single blended rate; the point is a real, non-zero spend figure on the page.
_COST_IN = float(os.environ.get("ALIBI_COST_IN_PER_M", "0.15")) / 1e6
_COST_OUT = float(os.environ.get("ALIBI_COST_OUT_PER_M", "0.60")) / 1e6
SPEND: dict = {"usd": 0.0, "in": 0, "out": 0, "calls": 0, "days": {}}


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
    for old in sorted(SPEND["days"])[:-30]:
        del SPEND["days"][old]


def _gate() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(_CONCURRENCY)
    return _sem


def enabled() -> bool:
    return bool(_KEY or _KEY2)


async def _ask(h, key, url, model, system, user, temperature):
    if not key:
        return ""
    r = await h.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    if r.status_code == 429:
        raise _RateLimited(float(r.headers.get("retry-after", "0") or 0))
    if r.status_code >= 300:
        raise httpx.HTTPStatusError("bad status", request=r.request, response=r)
    data = r.json()
    _account(data.get("usage") or {})
    return data["choices"][0]["message"]["content"]


class _RateLimited(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after


async def complete(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.7,
    timeout: float = 30.0,
) -> str:
    primary = model or MODEL  # per-agent override (a distinct model per role/crew member)
    # chain: the agent's model → the default Groq model (covers a bad/deprecated id) → the fallback provider
    chain = [(_KEY, _URL, primary), (_KEY, _URL, MODEL), (_KEY2, _URL2, MODEL2)]
    async with _gate(), httpx.AsyncClient(timeout=timeout) as h:
        for attempt in range(_RETRIES):
            for key, url, mdl in chain:
                try:
                    out = await _ask(h, key, url, mdl, system, user, temperature)
                    if out:
                        return out
                except _RateLimited as e:
                    await asyncio.sleep(max(e.retry_after, 0.6 * (2**attempt)))
                    break  # back off, then retry the whole provider chain
                except Exception:
                    continue
    return ""


async def complete_many(jobs: list[tuple[str, str, str]], temperature: float = 0.7) -> list[str]:
    # jobs = [(system, user, model), ...]; the semaphore in complete() bounds real concurrency
    return await asyncio.gather(*(complete(s, u, m, temperature) for s, u, m in jobs))


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else None
    except Exception:
        return None
