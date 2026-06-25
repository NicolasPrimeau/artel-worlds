from __future__ import annotations

import logging
import os

import httpx

from . import env

# Artel integration for VibeQuest.
# memories  → context fed to LLM before resolving each card
# tasks     → quest steps (created at quest start, claimed/completed by AI party)
# messages  → in-the-moment party coordination

log = logging.getLogger("vibequest.artel")

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
PROJECT = "vibequest"
_API_KEY = env("ARTEL_KEY")

_http: httpx.AsyncClient | None = None


def enabled() -> bool:
    return bool(_API_KEY)


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=8.0)
    return _http


def _headers() -> dict:
    return {"X-API-Key": _API_KEY, "Content-Type": "application/json"}


async def search_memory(query: str, limit: int = 5) -> str:
    if not enabled():
        return ""
    try:
        r = await _client().get(
            f"{ARTEL_URL}/memory/search",
            params={"q": query, "project": PROJECT, "limit": limit},
            headers=_headers(),
        )
        if r.status_code != 200:
            return ""
        entries = r.json()
        return "\n".join(e.get("content", "") for e in entries[:limit])
    except Exception as exc:
        log.debug("memory search failed: %s", exc)
        return ""


async def write_memory(content: str, tags: list[str] | None = None) -> None:
    if not enabled():
        return
    try:
        await _client().post(
            f"{ARTEL_URL}/memory",
            json={"content": content, "project": PROJECT, "tags": tags or []},
            headers=_headers(),
        )
    except Exception as exc:
        log.debug("memory write failed: %s", exc)


async def create_task(title: str, description: str) -> str | None:
    if not enabled():
        return None
    try:
        r = await _client().post(
            f"{ARTEL_URL}/tasks",
            json={"title": title, "description": description, "project": PROJECT},
            headers=_headers(),
        )
        if r.status_code == 200:
            return r.json().get("id")
    except Exception as exc:
        log.debug("task create failed: %s", exc)
    return None


async def complete_task(task_id: str, outcome: str) -> None:
    if not enabled():
        return
    try:
        await _client().post(
            f"{ARTEL_URL}/tasks/{task_id}/complete",
            json={"outcome": outcome},
            headers=_headers(),
        )
    except Exception as exc:
        log.debug("task complete failed: %s", exc)


async def send_message(from_agent: str, to_agent: str, content: str) -> None:
    if not enabled():
        return
    try:
        await _client().post(
            f"{ARTEL_URL}/messages",
            json={"from": from_agent, "to": to_agent, "content": content, "project": PROJECT},
            headers=_headers(),
        )
    except Exception as exc:
        log.debug("message send failed: %s", exc)
