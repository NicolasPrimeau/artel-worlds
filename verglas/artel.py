from __future__ import annotations

import logging
import os

from . import env

import httpx

# The meeting's bus IS Artel. Every line a survivor speaks and every vote it casts is broadcast to the
# verglas project on a real Artel server, as the seat agent — so the deduction literally happens over the
# coordination layer the worlds exist to show off. Off (a no-op) unless agent identities are configured,
# so local runs and an unprovisioned deploy still render from the in-memory transcript; best-effort
# throughout, so a slow or failed post never stalls the game.

log = logging.getLogger("verglas.artel")

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
PROJECT = env("PROJECT", "verglas")
_IDS = [s.strip() for s in env("AGENT_IDS").split(",") if s.strip()]
_KEYS = [s.strip() for s in env("AGENT_KEYS").split(",") if s.strip()]
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


async def say(index: int, name: str, text: str, subject: str = "verglas") -> None:
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


async def dm(from_index: int, to_index: int, text: str, subject: str = "verglas-whisper") -> None:
    # a PRIVATE agent-to-agent message on Artel (coalitions, "vote with me", "let's buddy up") — only
    # the two agents see it; the public viewer just knows a whisper happened.
    if not enabled() or not text:
        return
    sender, recipient = _seat(from_index), _seat(to_index)
    await _ensure_joined(sender)
    try:
        await _client().post(
            f"{ARTEL_URL}/messages",
            headers=_headers(sender),
            json={"to": recipient["id"], "subject": subject, "body": text[:280]},
        )
    except Exception as e:
        log.warning("artel dm failed for %s: %s", sender["id"], e)


async def clear_project() -> None:
    # wipe the project's tasks + messages so each game starts on a clean Artel board (owner-only — the
    # first seat created the project on join, so it owns it).
    if not enabled():
        return
    agent = AGENTS[0]
    await _ensure_joined(agent)
    try:
        await _client().post(
            f"{ARTEL_URL}/projects/{PROJECT}/clear",
            headers=_headers(agent),
            json={"tasks": True, "messages": True},
        )
    except Exception as e:
        log.warning("artel clear failed: %s", e)


async def create_task(title: str) -> str | None:
    # a task lit up on the board → a real open Artel task (created by the first seat, the "station")
    if not enabled():
        return None
    agent = AGENTS[0]
    await _ensure_joined(agent)
    try:
        r = await _client().post(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            json={"title": title[:140], "project": PROJECT, "priority": "low"},
        )
        return r.json().get("id") if r.status_code < 300 else None
    except Exception as e:
        log.warning("artel create_task failed: %s", e)
        return None


async def list_open_tasks() -> list[dict]:
    # read the live board straight off Artel — the open tasks ARE the source of truth for what's claimable
    if not enabled():
        return []
    agent = AGENTS[0]
    await _ensure_joined(agent)
    try:
        r = await _client().get(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            params={"project": PROJECT, "status": "open"},
        )
        return r.json() if r.status_code < 300 else []
    except Exception as e:
        log.warning("artel list_open_tasks failed: %s", e)
        return []


async def claim_task(index: int, task_id: str) -> bool:
    # returns True only if THIS agent won the claim — Artel answers 409 if another seat got there first,
    # so the claim is the real contention arbiter, not a local flag.
    if not enabled() or not task_id:
        return False
    agent = _seat(index)
    await _ensure_joined(agent)
    try:
        r = await _client().post(
            f"{ARTEL_URL}/tasks/{task_id}/claim", headers=_headers(agent), json={}
        )
        return r.status_code < 300
    except Exception as e:
        log.warning("artel claim failed for %s: %s", agent["id"], e)
        return False


async def complete_task(index: int, task_id: str) -> None:
    if not enabled() or not task_id:
        return
    agent = _seat(index)
    try:
        await _client().post(
            f"{ARTEL_URL}/tasks/{task_id}/complete", headers=_headers(agent), json={}
        )
    except Exception as e:
        log.warning("artel complete failed for %s: %s", agent["id"], e)


async def aclose() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
