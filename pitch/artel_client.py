from __future__ import annotations

import os

import httpx

# Thin async client for the Artel coordination server, reused across the line agents. Every call is
# best-effort: if Artel is unreachable or unconfigured it returns a falsy default and never raises,
# so a coordination hiccup can never stall or crash a live match. Auth is the same header pair the
# other worlds use (X-Agent-Id / X-Api-Key). Identities + URL come from the environment.

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
PITCH_PROJECT = os.environ.get("PITCH_PROJECT", "pitch")
_AGENT_IDS = [a for a in os.environ.get("PITCH_AGENT_IDS", "").split(",") if a.strip()]
_AGENT_KEYS = [k for k in os.environ.get("PITCH_AGENT_KEYS", "").split(",") if k.strip()]

# map a line role -> (agent_id, api_key); falls back to the first identity, or to local-only mode
_AGENTS: dict[str, dict] = {}
for _i, _role in enumerate(("captain", "def", "mid", "fwd")):
    if _i < len(_AGENT_IDS) and _i < len(_AGENT_KEYS):
        _AGENTS[_role] = {"id": _AGENT_IDS[_i], "key": _AGENT_KEYS[_i]}


def configured() -> bool:
    return bool(_AGENTS)


def _headers(role: str) -> dict | None:
    a = _AGENTS.get(role) or next(iter(_AGENTS.values()), None)
    if not a:
        return None
    return {"X-Agent-Id": a["id"], "X-Api-Key": a["key"], "content-type": "application/json"}


class Artel:
    def __init__(self, timeout: float = 4.0) -> None:
        self._http = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _req(self, role: str, method: str, path: str, **kw):
        h = _headers(role)
        if h is None:
            return None
        try:
            r = await self._http.request(method, f"{ARTEL_URL}{path}", headers=h, **kw)
            return r.json() if r.status_code < 300 else None
        except Exception:
            return None

    async def write_memory(self, role: str, content: str, tags: list[str]) -> dict | None:
        return await self._req(
            role,
            "POST",
            "/memory",
            json={"content": content, "project": PITCH_PROJECT, "tags": tags},
        )

    async def search_memory(self, role: str, q: str, tag: str = "", limit: int = 5) -> list:
        params = {"q": q, "project": PITCH_PROJECT, "limit": limit}
        if tag:
            params["tag"] = tag
        return await self._req(role, "GET", "/memory/search", params=params) or []

    async def emit_event(self, role: str, etype: str, payload: dict) -> dict | None:
        return await self._req(role, "POST", "/events", json={"type": etype, "payload": payload})

    async def create_task(self, role: str, title: str, tags: list[str]) -> dict | None:
        return await self._req(
            role,
            "POST",
            "/tasks",
            json={"title": title, "project": PITCH_PROJECT, "tags": tags},
        )

    async def claim_task(self, role: str, task_id: str) -> dict | None:
        return await self._req(role, "POST", f"/tasks/{task_id}/claim", json={"body": ""})

    async def list_tasks(self, role: str, tag: str = "", status: str = "open", limit: int = 20):
        params = {"project": PITCH_PROJECT, "status": status, "limit": limit}
        if tag:
            params["tag"] = tag
        return await self._req(role, "GET", "/tasks", params=params) or []

    async def clear_memory(self, role: str) -> None:
        # wipe the project's ephemeral coordination memory (owner-only). Result events are NOT
        # cleared, so the match record persists while the per-game line chatter is swept.
        await self._req(
            role,
            "POST",
            f"/projects/{PITCH_PROJECT}/clear",
            json={"memory": True, "messages": True},
        )
