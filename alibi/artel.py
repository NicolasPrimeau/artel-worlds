from __future__ import annotations

import logging
import os

import httpx

# The meeting's bus IS Artel. Every line a survivor speaks and every vote it casts is broadcast to the
# alibi project on a real Artel server, as the seat agent — so the deduction literally happens over the
# coordination layer the worlds exist to show off. Off (a no-op) unless agent identities are configured,
# so local runs and an unprovisioned deploy still render from the in-memory transcript; best-effort
# throughout, so a slow or failed post never stalls the game.

log = logging.getLogger("alibi.artel")

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
PROJECT = os.environ.get("ALIBI_PROJECT", "alibi")
_IDS = [s.strip() for s in os.environ.get("ALIBI_AGENT_IDS", "").split(",") if s.strip()]
_KEYS = [s.strip() for s in os.environ.get("ALIBI_AGENT_KEYS", "").split(",") if s.strip()]
AGENTS = [{"id": i, "key": k} for i, k in zip(_IDS, _KEYS)]

_http: httpx.AsyncClient | None = None
_joined: set[str] = set()


def enabled() -> bool:
    return bool(AGENTS)


def status() -> dict:
    return {"enabled": enabled(), "url": ARTEL_URL, "project": PROJECT, "seats": len(AGENTS)}


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(8.0))
    return _http


def _headers(agent: dict) -> dict:
    return {
        "X-Agent-Id": agent["id"],
        "X-Api-Key": agent["key"],
        "content-type": "application/json",
    }


def _seat(index: int) -> dict:
    return AGENTS[index % len(AGENTS)]


async def _ensure_joined(agent: dict) -> None:
    if agent["id"] in _joined:
        return
    _joined.add(agent["id"])  # mark first so a failure doesn't make us hammer join every message
    try:
        await _client().post(f"{ARTEL_URL}/projects/{PROJECT}/join", headers=_headers(agent))
    except Exception as e:
        log.warning("artel join failed for %s: %s", agent["id"], e)


async def say(index: int, name: str, text: str, subject: str = "alibi") -> None:
    if not enabled() or not text:
        return
    agent = _seat(index)
    await _ensure_joined(agent)
    try:
        await _client().post(
            f"{ARTEL_URL}/messages",
            headers=_headers(agent),
            json={"to": f"project:{PROJECT}", "subject": subject, "body": f"{name}: {text}"[:280]},
        )
    except Exception as e:
        log.warning("artel say failed for %s: %s", agent["id"], e)


async def aclose() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
