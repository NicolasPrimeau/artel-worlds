from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

from .incidents import Incident

log = logging.getLogger("watchtower")

# Watchtower's responders are real LLM agents. Two fleets run the SAME model and the SAME prompt and
# the SAME tools; the ONLY thing that differs is where their runbooks live. The Artel fleet's recall/
# remember go over HTTP to a shared artel.run project — any responder can apply a runbook any other
# responder wrote. The solo fleet's notes live in a per-agent in-process store that is never shared.
# Hold everything else equal and the gap that opens is the value of SHARING, nothing else.

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
WATCHTOWER_PROJECT = os.environ.get("WATCHTOWER_PROJECT", "watchtower")
PROVIDER = os.environ.get("WATCHTOWER_LLM_PROVIDER", "openai")
MODEL = os.environ.get("WATCHTOWER_MODEL", "gemini-2.5-flash-lite")
LLM_KEY = os.environ.get("WATCHTOWER_LLM_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
_DEFAULT_URL = (
    "https://api.anthropic.com/v1/messages"
    if PROVIDER == "anthropic"
    else "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_URL = os.environ.get("WATCHTOWER_LLM_URL", _DEFAULT_URL)
LLM_VERSION = os.environ.get("WATCHTOWER_LLM_VERSION", "2023-06-01")
REASONING = os.environ.get("WATCHTOWER_REASONING", "none")
SPEND_CAP_DAILY_USD = float(os.environ.get("WATCHTOWER_SPEND_CAP_DAILY_USD", "0.50"))
MAX_ROUNDS = int(os.environ.get("WATCHTOWER_MAX_ROUNDS", "16"))
LLM_TIMEOUT = float(os.environ.get("WATCHTOWER_LLM_TIMEOUT", "20"))


def _make_ep(provider, model, url, key, version, cin, cout):
    return {
        "provider": provider,
        "model": model,
        "url": url,
        "key": key,
        "version": version,
        "cin": float(cin) / 1_000_000,
        "cout": float(cout) / 1_000_000,
    }


# Primary driver plus a second provider used only when the primary rate-limits or errors — a 429 on
# one fails the decision over to another LLM instead of leaving an incident unworked. Same failover
# spine as Phalanx; set WATCHTOWER_LLM2_KEY to arm the fallback.
PRIMARY = _make_ep(
    PROVIDER,
    MODEL,
    LLM_URL,
    LLM_KEY,
    LLM_VERSION,
    os.environ.get("WATCHTOWER_COST_IN", "0.10"),
    os.environ.get("WATCHTOWER_COST_OUT", "0.40"),
)
_LLM2_KEY = os.environ.get("WATCHTOWER_LLM2_KEY", "")
FALLBACK = (
    _make_ep(
        os.environ.get("WATCHTOWER_LLM2_PROVIDER", "openai"),
        os.environ.get("WATCHTOWER_LLM2_MODEL", "gemini-2.5-flash"),
        os.environ.get(
            "WATCHTOWER_LLM2_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        _LLM2_KEY,
        os.environ.get("WATCHTOWER_LLM2_VERSION", "2023-06-01"),
        os.environ.get("WATCHTOWER_LLM2_COST_IN", "0.30"),
        os.environ.get("WATCHTOWER_LLM2_COST_OUT", "2.50"),
    )
    if _LLM2_KEY
    else None
)
_ENDPOINTS = [PRIMARY] + ([FALLBACK] if FALLBACK else [])

# Identical for both fleets — fairness depends on it. The prompt teaches the loop (recall, diagnose,
# minimal fix, record a generalizable runbook) but nothing fleet-specific; the only variable is
# whether the recall/remember tools reach a shared store or a private one.
SYSTEM = (
    "You are an on-call site reliability engineer for a 10-node service (lb, web, api, auth, db, "
    "db-replica, cache, queue, worker-1, worker-2). A monitoring page just fired. Your job is to "
    "restore the service as FAST as possible — every action you take costs real minutes, so wasted "
    "steps are the whole cost of an incident.\n"
    "THE GRAPH (who depends on whom): lb->web->api->{db,cache,queue,auth}; auth->db; "
    "db-replica->db; worker-1,worker-2->{queue,db}. A sick node makes everything ABOVE it look "
    "sick too — the loudest alarm is usually a symptom, not the cause. Find the real root before "
    "you remediate.\n"
    "YOUR LOOP, in order:\n"
    "1. RECALL FIRST: call recall with the symptoms. If a past runbook matches, follow it — that is "
    "the entire point of having one. Do not re-derive what you already know.\n"
    "1b. CHECK THE BOARD: call check_board once — an open task may be a past UNRESOLVED incident of "
    "this same fault with notes from whoever tried before. If your fix cracks an open board item, "
    "close_task it.\n"
    "2. DIAGNOSE: inspect / read_logs the suspect nodes to confirm the root cause. Cheap, but not "
    "free — don't inspect the whole graph when the runbook already named the fix.\n"
    "3. REMEDIATE: apply the SINGLE correct fix (sometimes an ordered pair). A wrong or needless "
    "remediation costs extra and changes nothing. Available: restart, scale, rollback, clear_queue, "
    "failover, rotate — each on a named node.\n"
    "4. RECORD: once resolved, call remember with a runbook that will help NEXT time — state the "
    "root cause and the fix in terms of the SYMPTOM PATTERN (kind of node, which dials, the order of "
    "steps), NOT this incident's specific node name or numbers (those change every time). One tight "
    "rule. Skip platitudes you already know.\n"
    "You may also file_task work that should NOT block restoring service (preventive fixes, "
    "monitoring gaps, cleanup you noticed). Keep acting until the incident reads RESOLVED."
)

_NODE_PROP = {
    "type": "string",
    "description": "target node: lb, web, api, auth, db, db-replica, cache, queue, worker-1, worker-2",
}
TOOLS = [
    {
        "name": "recall",
        "description": "Search your runbooks for a past incident whose symptoms match this one. Call "
        "this FIRST, before inspecting — a matching runbook is the fastest path to the fix.",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the symptoms in a few words"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "inspect",
        "description": "Read a node's live status, metrics, dependencies, and anomalies.",
        "schema": {"type": "object", "properties": {"node": _NODE_PROP}, "required": ["node"]},
    },
    {
        "name": "read_logs",
        "description": "Read a node's recent log lines — the error signatures that name a root cause.",
        "schema": {"type": "object", "properties": {"node": _NODE_PROP}, "required": ["node"]},
    },
    {
        "name": "remediate",
        "description": "Apply a remediation to a node. action is one of restart, scale, rollback, "
        "clear_queue, failover, rotate. The wrong action, or the right one on the wrong node, costs "
        "time and fixes nothing. Some incidents need two in a specific order.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["restart", "scale", "rollback", "clear_queue", "failover", "rotate"],
                },
                "node": _NODE_PROP,
            },
            "required": ["action", "node"],
        },
    },
    {
        "name": "remember",
        "description": "Save ONE runbook after resolving: the symptom pattern -> root cause -> the "
        "ordered fix, generalized (node KIND and dials, not this node's name/numbers). For next time.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "check_board",
        "description": "List the team's open tasks. Past UNRESOLVED incidents sit here with notes — "
        "if this fault looks familiar, the board may say what was already tried and failed.",
        "schema": {"type": "object", "properties": {}},
    },
    {
        "name": "file_task",
        "description": "File a task for work that should not block restoring service: a preventive "
        "fix, a monitoring gap, cleanup. Someone on the team picks it up later.",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "short imperative title"},
                "detail": {"type": "string", "description": "what and why, one or two lines"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "close_task",
        "description": "Complete an open board task by id (from check_board) — use when your fix "
        "resolves what the task tracked.",
        "schema": {
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
    },
]


def _build_payload(ep, system, transcript, force=None):
    if ep["provider"] == "anthropic":
        messages = []
        for e in transcript:
            if e["role"] == "user":
                messages.append({"role": "user", "content": e["text"]})
            elif e["role"] == "assistant":
                blocks = [{"type": "text", "text": e["text"]}] if e.get("text") else []
                blocks += [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]}
                    for c in e["calls"]
                ]
                messages.append({"role": "assistant", "content": blocks})
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"]}
                            for r in e["results"]
                        ],
                    }
                )
        payload = {
            "model": ep["model"],
            "max_tokens": 400,
            "system": system,
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
                for t in TOOLS
            ],
            "tool_choice": {"type": "any"},
            "messages": messages,
        }
        headers = {
            "x-api-key": ep["key"],
            "anthropic-version": ep["version"],
            "content-type": "application/json",
        }
        return ep["url"], payload, headers

    messages = [{"role": "system", "content": system}]
    for e in transcript:
        if e["role"] == "user":
            messages.append({"role": "user", "content": e["text"]})
        elif e["role"] == "assistant":
            messages.append(
                {
                    "role": "assistant",
                    "content": e.get("text") or None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["input"])},
                        }
                        for c in e["calls"]
                    ],
                }
            )
        else:
            for r in e["results"]:
                messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"]})
    payload = {
        "model": ep["model"],
        "max_tokens": 400,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["schema"],
                },
            }
            for t in TOOLS
        ],
        "tool_choice": "required",
        "messages": messages,
    }
    if REASONING and "gemini" in ep["model"]:
        payload["reasoning_effort"] = REASONING
    headers = {"authorization": f"Bearer {ep['key']}", "content-type": "application/json"}
    return ep["url"], payload, headers


def _parse(ep, data):
    if ep["provider"] == "anthropic":
        usage = data.get("usage", {})
        text, calls = "", []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    }
                )
        return text, calls, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    usage = data.get("usage", {})
    msg = (data.get("choices") or [{}])[0].get("message", {})
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function", {})
        try:
            inp = json.loads(fn.get("arguments") or "{}")
        except Exception:
            inp = {}
        calls.append({"id": c.get("id"), "name": fn.get("name"), "input": inp})
    return (
        msg.get("content") or "",
        calls,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
    )


THROTTLED: dict[str, int] = {}
_down_until: dict[str, float] = {}
_DOWN_DAILY = 900.0
_DOWN_BURST = 20.0


def _mark_throttled(ep, body):
    THROTTLED[ep["model"]] = THROTTLED.get(ep["model"], 0) + 1
    cooldown = _DOWN_DAILY if ("per day" in body or "RPD" in body) else _DOWN_BURST
    _down_until[ep["model"]] = asyncio.get_event_loop().time() + cooldown


def _live_endpoints():
    now = asyncio.get_event_loop().time()
    eps = [ep for ep in _ENDPOINTS if ep["key"] and _down_until.get(ep["model"], 0) <= now]
    return eps or [ep for ep in _ENDPOINTS if ep["key"]]


async def _chat(http, system, transcript):
    last = None
    for ep in _live_endpoints():
        url, payload, headers = _build_payload(ep, system, transcript)
        r = await http.post(url, headers=headers, json=payload)
        if r.status_code < 300:
            text, calls, tin, tout = _parse(ep, r.json())
            return text, calls, tin, tout, ep
        if r.status_code == 429:
            _mark_throttled(ep, r.text[:300])
        log.warning("LLM %s HTTP %s: %s", ep["model"], r.status_code, r.text[:200])
        last = (ep, r)
    if last is None:
        raise RuntimeError("no llm endpoint configured")
    raise RuntimeError(f"llm http {last[1].status_code}")


def _headers(agent):
    return {
        "X-Agent-Id": agent["id"],
        "X-Api-Key": agent["key"],
        "content-type": "application/json",
    }


def has_llm() -> bool:
    return bool(LLM_KEY) or bool(FALLBACK and FALLBACK["key"])


# --- runbook backends: identical interface, opposite visibility. THIS is the experiment. ---
class ArtelStore:
    """Shared runbooks over artel.run. Every Artel-fleet responder reads and writes ONE project, so a
    runbook earned by one is instantly usable by all three — and the archivist can promote the stable
    ones to docs. Messages land in the project feed: the live war room."""

    def __init__(self, http: httpx.AsyncClient, agent: dict):
        self.http = http
        self.agent = agent
        self.shared = True
        self.feed: list[dict] = []

    async def join(self) -> None:
        try:
            await self.http.post(
                f"{ARTEL_URL}/projects/{WATCHTOWER_PROJECT}/join", headers=_headers(self.agent)
            )
        except Exception:
            pass

    async def recall(self, query: str) -> str:
        if not query:
            return ""
        try:
            r = await self.http.get(
                f"{ARTEL_URL}/memory/search",
                headers=_headers(self.agent),
                params={"q": query, "project": WATCHTOWER_PROJECT, "limit": 4},
            )
            rows = r.json() if r.status_code < 300 else []
        except Exception:
            return ""
        if not isinstance(rows, list):
            return ""
        return " | ".join(m.get("content", "")[:240] for m in rows if m.get("content"))

    async def remember(self, text: str) -> None:
        if not text:
            return
        try:
            await self.http.post(
                f"{ARTEL_URL}/memory",
                headers=_headers(self.agent),
                json={"content": text[:400], "project": WATCHTOWER_PROJECT, "tags": ["runbook"]},
            )
            self._activity(f"saved runbook: {text[:90]}")
        except Exception:
            pass

    async def _open_tasks(self) -> list[dict]:
        r = await self.http.get(
            f"{ARTEL_URL}/tasks",
            headers=_headers(self.agent),
            params={"project": WATCHTOWER_PROJECT, "status": "open"},
        )
        rows = r.json() if r.status_code < 300 else []
        return rows if isinstance(rows, list) else []

    async def board(self) -> str:
        try:
            rows = await self._open_tasks()
        except Exception:
            return ""
        return "\n".join(
            f"[{t['id'][:8]}] {t.get('title', '')[:90]} — {t.get('description', '')[:140]}"
            for t in rows[:6]
        )

    async def file_task(self, title: str, detail: str, tags: list[str] | None = None) -> None:
        if not title:
            return
        try:
            await self.http.post(
                f"{ARTEL_URL}/tasks",
                headers=_headers(self.agent),
                json={
                    "title": title[:120],
                    "description": (detail or "")[:400],
                    "project": WATCHTOWER_PROJECT,
                    "tags": tags or [],
                },
            )
            self._activity(f"filed task: {title[:80]}")
        except Exception:
            pass

    async def finish_task(self, task_id: str) -> bool:
        if not task_id:
            return False
        try:
            h = _headers(self.agent)
            full = next(
                (t["id"] for t in await self._open_tasks() if t["id"].startswith(task_id)), None
            )
            if not full:
                return False
            await self.http.post(f"{ARTEL_URL}/tasks/{full}/claim", headers=h)
            r = await self.http.post(f"{ARTEL_URL}/tasks/{full}/complete", headers=h)
            if r.status_code < 300:
                self._activity(f"completed task {task_id[:8]}")
            return r.status_code < 300
        except Exception:
            return False

    async def open_incident(self, seq: int, title: str, family: str, alert: str) -> str | None:
        # every incident is a real task on the shared board: filed and claimed by the responder,
        # completed on resolution, left open with a note on a miss — the board IS the incident log
        try:
            h = _headers(self.agent)
            r = await self.http.post(
                f"{ARTEL_URL}/tasks",
                headers=h,
                json={
                    "title": f"Incident #{seq}: {title}",
                    "description": alert[:300],
                    "project": WATCHTOWER_PROJECT,
                    "tags": [family, "incident"],
                },
            )
            if r.status_code >= 300:
                return None
            tid = r.json().get("id")
            await self.http.post(f"{ARTEL_URL}/tasks/{tid}/claim", headers=h)
            self._activity(f"claimed incident #{seq}: {title[:70]}")
            return tid
        except Exception:
            return None

    async def close_incident(self, task_id: str | None, resolved: bool, note: str) -> None:
        if not task_id:
            return
        try:
            h = _headers(self.agent)
            if resolved:
                await self.http.post(f"{ARTEL_URL}/tasks/{task_id}/complete", headers=h)
                self._activity("resolved — incident task completed")
            else:
                await self.http.post(
                    f"{ARTEL_URL}/tasks/{task_id}/comments", headers=h, json={"body": note[:300]}
                )
                await self.http.post(f"{ARTEL_URL}/tasks/{task_id}/unclaim", headers=h)
                self._activity("unresolved — left open on the board with a handoff note")
        except Exception:
            pass

    async def sweep_family(self, family: str) -> None:
        # the family just got cracked: complete any older incident tasks still open for it
        try:
            for t in await self._open_tasks():
                if family in (t.get("tags") or []) and "incident" in (t.get("tags") or []):
                    h = _headers(self.agent)
                    await self.http.post(f"{ARTEL_URL}/tasks/{t['id']}/claim", headers=h)
                    await self.http.post(f"{ARTEL_URL}/tasks/{t['id']}/complete", headers=h)
        except Exception:
            pass

    async def save_handoff(self, summary: str) -> None:
        try:
            await self.http.post(
                f"{ARTEL_URL}/sessions/handoff",
                headers=_headers(self.agent),
                json={"summary": summary[:300]},
            )
        except Exception:
            pass

    async def handoff(self) -> str:
        # the native Artel session mechanism: my own last shift summary, plus every runbook the
        # TEAM added while I was off rotation (the memory delta) — that delta is the sharing edge
        try:
            r = await self.http.get(f"{ARTEL_URL}/sessions/handoff", headers=_headers(self.agent))
            data = r.json() if r.status_code < 300 else {}
        except Exception:
            return ""
        lines = []
        last = data.get("last_handoff") or {}
        if last.get("summary"):
            lines.append(f"your last shift: {last['summary'][:200]}")
        delta = [
            m for m in data.get("memory_delta") or [] if m.get("project") == WATCHTOWER_PROJECT
        ]
        if delta:
            lines.append(f"while you were off shift the team added {len(delta)} runbook(s):")
            lines += [f"- {m.get('content', '')[:140]}" for m in delta[-4:]]
        return "\n".join(lines)

    def _activity(self, text: str) -> None:
        self.feed.append({"from": self.agent["id"], "text": text[:200]})
        del self.feed[:-40]


def _tokens(s: str) -> set[str]:
    return {w for w in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(w) > 2}


class SoloStore:
    """A single responder's private notebook — never shared, never pooled. Same recall/remember
    interface as the Artel store, but a runbook one solo responder writes is invisible to the other
    two: each must rediscover every fault family on its own. The control arm of the experiment."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.shared = False
        self.notes: list[str] = []
        self.feed: list[dict] = []
        self.tasks: list[dict] = []
        self.last_shift: str = ""

    async def join(self) -> None:
        return None

    async def recall(self, query: str) -> str:
        if not query or not self.notes:
            return ""
        q = _tokens(query)
        scored = sorted(self.notes, key=lambda n: len(q & _tokens(n)), reverse=True)
        top = [n for n in scored if q & _tokens(n)][:3]
        return " | ".join(n[:240] for n in top)

    async def remember(self, text: str) -> None:
        if text:
            self.notes.append(text[:400])
            del self.notes[:-200]

    async def save_handoff(self, summary: str) -> None:
        self.last_shift = summary[:300]

    async def handoff(self) -> str:
        return f"your last shift: {self.last_shift}" if self.last_shift else ""

    async def board(self) -> str:
        return "\n".join(
            f"[{t['id']}] {t['title'][:90]} — {t['detail'][:140]}" for t in self.tasks if t["open"]
        )[:1200]

    async def file_task(self, title: str, detail: str, tags: list[str] | None = None) -> None:
        if title:
            self.tasks.append(
                {
                    "id": f"t{len(self.tasks)}",
                    "title": title,
                    "detail": detail or "",
                    "tags": tags or [],
                    "open": True,
                }
            )
            del self.tasks[:-50]

    async def finish_task(self, task_id: str) -> bool:
        for t in self.tasks:
            if t["id"].startswith(task_id) and t["open"]:
                t["open"] = False
                return True
        return False

    async def open_incident(self, seq: int, title: str, family: str, alert: str) -> str | None:
        await self.file_task(f"Incident #{seq}: {title}", alert, [family, "incident"])
        return self.tasks[-1]["id"]

    async def close_incident(self, task_id: str | None, resolved: bool, note: str) -> None:
        for t in self.tasks:
            if t["id"] == task_id:
                if resolved:
                    t["open"] = False
                else:
                    t["detail"] = (t["detail"] + " | " + note)[:400]

    async def sweep_family(self, family: str) -> None:
        for t in self.tasks:
            if t["open"] and family in t.get("tags", []):
                t["open"] = False


def _render_incident(inc: Incident) -> str:
    lines = [
        f"INCIDENT #{inc.seq} — {inc.spec.alert}",
        "Monitoring board: " + (", ".join(inc.infra.alarms()) or "all green"),
        "Resolve it. Recall a runbook first; inspect to confirm the root cause; apply the minimal "
        "fix; record a generalizable runbook when done.",
    ]
    return "\n".join(lines)


def _state_note(inc: Incident) -> str:
    board = ", ".join(inc.infra.alarms()) or "all green"
    status = "RESOLVED" if inc.resolved else ("CLOSED (unresolved)" if inc.missed else "still OPEN")
    return f"[board: {board}] [actions so far: {len(inc.actions)}] incident is {status}."


async def respond(http, inc: Incident, store, counts: dict | None = None, on_step=None) -> float:
    # one responder works one incident to resolution (or until the action cap closes it). Returns the
    # USD spent on LLM calls. The store is the only thing that differs between the two fleets.
    cost = 0.0
    recorded = False
    intro = _render_incident(inc)
    handoff = await store.handoff()
    if handoff:
        intro += "\nShift context:\n" + handoff
    transcript = [{"role": "user", "text": intro}]
    for _ in range(MAX_ROUNDS):
        try:
            text, calls, tin, tout, ep = await _chat(http, SYSTEM, transcript)
        except Exception as e:
            log.warning("watchtower responder %s llm failed: %s", inc.fleet, e)
            break
        cost += tin * ep["cin"] + tout * ep["cout"]
        if not calls:
            break
        transcript.append({"role": "assistant", "text": text, "calls": calls})
        results = []
        for c in calls:
            name = c["name"]
            inp = c.get("input") or {}
            if counts is not None:
                counts[name] = counts.get(name, 0) + 1
            if name == "remediate":
                out = inc.act(str(inp.get("action", "")), str(inp.get("node", "")))
            elif name in ("inspect", "read_logs"):
                out = inc.act(name, str(inp.get("node", "")))
            elif name == "recall":
                found = await store.recall(str(inp.get("query", "")))
                out = {"runbooks": found} if found else {"runbooks": "none on file — you're first"}
            elif name == "remember":
                await store.remember(str(inp.get("text", "")))
                recorded = recorded or bool(inp.get("text"))
                out = {"result": "runbook saved"}
            elif name == "check_board":
                b = await store.board()
                out = {"board": b or "board is clear"}
            elif name == "file_task":
                await store.file_task(str(inp.get("title", "")), str(inp.get("detail", "")))
                out = {"result": "task filed"}
            elif name == "close_task":
                done = await store.finish_task(str(inp.get("task_id", "")))
                out = {"result": "task completed" if done else "no open task with that id"}
            else:
                out = {"error": f"unknown tool {name}"}
            results.append({"id": c["id"], "output": json.dumps(out)})
        transcript.append({"role": "tool", "results": results})
        if on_step:
            await on_step()
        if inc.resolved or inc.missed:
            break
        transcript.append({"role": "user", "text": _state_note(inc)})
    if inc.resolved and not recorded:
        # the resolving action ends the loop before the model's RECORD step would run — without
        # this round no fleet ever writes a runbook and the whole experiment runs unshared
        cost += await _record_runbook(http, inc, store, transcript, counts)
    return cost


async def _record_runbook(http, inc: Incident, store, transcript, counts) -> float:
    transcript.append(
        {
            "role": "user",
            "text": "Resolved. Before you go: ONE remember call — the alert symptoms, the root "
            "cause, and the exact fix that worked, written so the next responder can apply it "
            "without diagnosing.",
        }
    )
    cost = 0.0
    try:
        text, calls, tin, tout, ep = await _chat(http, SYSTEM, transcript)
        cost = tin * ep["cin"] + tout * ep["cout"]
        for c in calls:
            if c["name"] == "remember":
                t = str((c.get("input") or {}).get("text", ""))
                if t:
                    await store.remember(t)
                    if counts is not None:
                        counts["remember"] = counts.get("remember", 0) + 1
                    return cost
    except Exception as e:
        log.warning("watchtower runbook round failed: %s", e)
    # model skipped it — synthesize the runbook from the fix that actually worked (identical
    # fallback for both fleets, so fairness holds)
    steps = " then ".join(f"{a} {n}" for a, n in inc.spec.fix)
    await store.remember(f"runbook {inc.spec.family}: alert '{inc.spec.alert}'. fix: {steps}.")
    return cost
