from __future__ import annotations

import asyncio
import json
import re
import logging
import os

import httpx

from .config import DEFAULT
from .control import STRATEGIES, Bot

log = logging.getLogger("phalanx")

# The Artel team is three real LLM agents — one per tank — each running its own loop
# against live artel.run. Nothing here decides strategy: every turn the model perceives
# its tank, reads what its teammates told it through Artel, and chooses an action. Any
# coordination is the models talking to each other over Artel. Red, by contrast, is the
# deterministic seek-and-destroy Bot. The whole demo is: same arena, same guns — the only
# thing Artel has that Red doesn't is each other, through Artel.
#
# The model is a pure env swap. PHALANX_LLM_PROVIDER=anthropic uses the Messages API
# (Haiku); =openai uses any OpenAI-compatible chat endpoint — Gemini Flash, Groq, DeepSeek,
# etc. — set PHALANX_LLM_URL, PHALANX_MODEL and PHALANX_LLM_KEY to point it anywhere cheaper.

ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
PHALANX_PROJECT = os.environ.get("PHALANX_PROJECT", "phalanx")
# Default is Gemini 2.5 Flash over Google's OpenAI-compatible endpoint. OpenAI is out: its
# 10k requests/day quota dies mid-stream under continuous play. The reflex floors in decide()
# cover Gemini's occasional tool-calling misses. Set PHALANX_LLM_PROVIDER=anthropic
# (+ ANTHROPIC_API_KEY) for Claude, or point PHALANX_LLM_URL / PHALANX_MODEL / PHALANX_LLM_KEY
# at any other OpenAI-compatible provider.
PROVIDER = os.environ.get("PHALANX_LLM_PROVIDER", "openai")
MODEL = os.environ.get("PHALANX_MODEL", "gemini-3.1-flash-lite")
LLM_KEY = os.environ.get("PHALANX_LLM_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
_DEFAULT_URL = (
    "https://api.anthropic.com/v1/messages"
    if PROVIDER == "anthropic"
    else "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_URL = os.environ.get("PHALANX_LLM_URL", _DEFAULT_URL)
LLM_VERSION = os.environ.get("PHALANX_LLM_VERSION", "2023-06-01")
SPEND_CAP_USD = float(os.environ.get("PHALANX_SPEND_CAP_USD", "20"))
REASONING = os.environ.get("PHALANX_REASONING", "none")  # gemini thinking effort; none = off


def _make_ep(provider: str, model: str, url: str, key: str, version: str, cin: str, cout: str):
    return {
        "provider": provider,
        "model": model,
        "url": url,
        "key": key,
        "version": version,
        "cin": float(cin) / 1_000_000,  # $/input token
        "cout": float(cout) / 1_000_000,  # $/output token
    }


# Primary driver, plus a SECOND provider used only when the primary rate-limits or errors — so a
# 429 on one provider fails the decision over to another LLM instead of dropping the tank to a
# scripted move. Blue stays model-driven end to end. Default fallback is Gemini Flash-Lite
# (cheapest) over Google's OpenAI-compatible endpoint; set PHALANX_LLM2_KEY to arm it.
PRIMARY = _make_ep(
    PROVIDER,
    MODEL,
    LLM_URL,
    LLM_KEY,
    LLM_VERSION,
    os.environ.get("PHALANX_COST_IN", "0.25"),
    os.environ.get("PHALANX_COST_OUT", "1.50"),
)
_LLM2_KEY = os.environ.get("PHALANX_LLM2_KEY", "")
FALLBACK = (
    _make_ep(
        os.environ.get("PHALANX_LLM2_PROVIDER", "openai"),
        os.environ.get("PHALANX_LLM2_MODEL", "gemini-3.5-flash"),
        os.environ.get(
            "PHALANX_LLM2_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        _LLM2_KEY,
        os.environ.get("PHALANX_LLM2_VERSION", "2023-06-01"),
        os.environ.get("PHALANX_LLM2_COST_IN", "1.50"),
        os.environ.get("PHALANX_LLM2_COST_OUT", "9.00"),
    )
    if _LLM2_KEY
    else None
)
_ENDPOINTS = [PRIMARY] + ([FALLBACK] if FALLBACK else [])
# Hard ceiling on a tank's WHOLE decision (all tool rounds together) so a slow model can never
# stall the synchronous tick — past it the tank falls back to a reflex move instead of freezing.
DECIDE_DEADLINE = float(os.environ.get("PHALANX_DECIDE_DEADLINE", "12"))
LLM_TIMEOUT = float(os.environ.get("PHALANX_LLM_TIMEOUT", "12"))

# Concise. Names the teammates, says coordinate through Artel — and stops there. No
# instructions on HOW to use Artel: the coordination has to emerge, not be scripted.
COMMAND_EVERY = int(os.environ.get("PHALANX_COMMAND_EVERY", "3"))  # ticks between routine orders

# The commander layer: each tank's MOTOR is the same deterministic Bot red uses (targeting,
# range bands, pathing, prudence — identical competence on both sides). The LLM agent COMMANDS
# its bot through standing orders, at command cadence, from what it learns over Artel. Red
# gets no orders: coordination is exactly and only what Artel buys.
SYSTEM = (
    "You command one tank in team Artel's three-tank unit in Phalanx (hex arena, shrinking "
    "safe zone, last team standing). Your teammates are {mates}; everything the unit shares "
    "flows through Artel. Your tank DRIVES ITSELF competently (targeting, range-keeping, "
    "pathing, retreating when hurt). You do not steer — you give STANDING ORDERS when the "
    "situation calls for them:\n"
    "- focus: concentrate the unit's fire on ONE enemy id (pass focus_at [q,r] from a "
    "teammate's report and your tank will hunt a target it has never seen — that is the "
    "whole point of the radio).\n"
    "- regroup [q,r]: rally there now. Use it to mass the unit, rescue a teammate under "
    "fire, or collapse on a kill.\n"
    "- post [q,r]: where to hold when it has no contact (ambush corners, cover the zone).\n"
    "- clear_orders: release your tank back to its own instincts.\n"
    "DOCTRINE, in order: a teammate UNDER FIRE outranks everything — focus their attacker "
    "and converge; mass fire on ONE enemy (a 3v1 wins, three 1v1s lose); fight INSIDE the "
    "shrinking zone, near cover, never strung out alone. A tank that holds still — not "
    "firing, untouched, in the zone — REPAIRS +2 energy per turn: post a hurt tank behind "
    "cover to recover while the others screen it, and keep pressure on hurt enemies so "
    "they never can.\n"
    "ARTEL, in the same call — say: report only NEW actionable facts with coordinates "
    "('SPOTTED #5 (7,4) energy 40', 'FOCUS #5', 'RALLY (8,6)'); teammates' reports are real "
    "positions you cannot see. objective: your ONE medium-term commitment on the team "
    "board; change it only when reality breaks it. lesson: save one concrete lesson when a "
    "call clearly won or lost a fight; [WIN]/[LOSS] lessons from past matches arrive in "
    "your context — trust wins.\n"
    "No orders needed? Send none — your tank fights fine alone; orders are for making three "
    "tanks fight as ONE."
)

SOLO_SYSTEM = (
    "You command one tank of team Red in Phalanx (hex arena, shrinking safe zone, last team "
    "standing). Your teammates are {mates}, but you have NO communication with them — no "
    "radio, no shared map. Your tank drives itself competently; you may give it standing "
    "orders from what YOU can see: focus (enemy id), regroup [q,r], post [q,r], "
    "clear_orders. No orders needed? Send none."
)

TOOLS = [
    {
        "name": "command",
        "description": "Your one call: standing orders for your tank (all optional — omit "
        "everything to leave it fighting on instinct) plus any Artel traffic.",
        "schema": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "integer",
                    "description": "enemy tank id the unit should concentrate on (0 = none)",
                },
                "focus_at": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[q,r] last reported position of the focus target — lets "
                    "your tank hunt an enemy it has not seen itself",
                },
                "regroup": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[q,r] rally point: the tank moves there now, then resumes",
                },
                "post": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[q,r] where to hold when it has no contact",
                },
                "clear_orders": {
                    "type": "boolean",
                    "description": "true = drop all standing orders, back to instinct",
                },
                "say": {
                    "type": "string",
                    "description": "optional Artel report to the team: NEW actionable facts "
                    "with coordinates only — silence beats noise",
                },
                "objective": {
                    "type": "string",
                    "description": "optional: replace YOUR objective on the team board",
                },
                "lesson": {
                    "type": "string",
                    "description": "optional: one lesson that will still be true next match",
                },
                "plan": {
                    "type": "string",
                    "description": "one short line of intent — shown back to you next time",
                },
            },
        },
    }
]
TOOLS_SOLO = [
    {
        "name": "command",
        "description": "Standing orders for your tank — all optional; omit everything to "
        "leave it fighting on instinct.",
        "schema": {
            "type": "object",
            "properties": {
                k: v
                for k, v in TOOLS[0]["schema"]["properties"].items()
                if k in ("focus", "regroup", "post", "clear_orders", "plan")
            },
        },
    }
]


def _hexdist(aq: int, ar: int, bq: int, br: int) -> int:
    dq, dr = aq - bq, ar - br
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


def _on_map(p: dict, q: int, r: int) -> bool:
    R = p.get("map_radius", (p.get("width", 15) - 1) // 2)
    return _hexdist(q, r, R, R) <= R


# Pointy-top hex axial directions, index == heading 0..5 (same convention as the arena).
AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def _open_dir(wallset: set, want: int) -> int:
    # deflect a desired step to the nearest direction not blocked by cover, so a tank routes
    # around an obstacle instead of nosing into it and getting stuck
    for off in (0, 1, -1, 2, -2, 3):
        d = (want + off) % 6
        if AXIAL_DIRS[d] not in wallset:
            return d
    return want


# --- provider adapters: a neutral transcript in/out, provider wire format hidden here ---
def _build_payload(
    ep: dict,
    system: str,
    transcript: list[dict],
    force_act: bool = False,
    tools: list | None = None,
) -> tuple[str, dict, dict]:
    toolset = tools if tools is not None else TOOLS
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
            "max_tokens": 320,
            "system": system,
            "tools": [
                {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
                for t in toolset
            ],
            "tool_choice": {"type": "tool", "name": toolset[0]["name"]}
            if force_act
            else {"type": "any"},
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
        "max_tokens": 320,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["schema"],
                },
            }
            for t in toolset
        ],
        "tool_choice": {"type": "function", "function": {"name": toolset[0]["name"]}}
        if force_act
        else "required",
        "messages": messages,
    }
    if REASONING and "gemini" in ep["model"]:
        # Gemini 2.5 models think by default and bill the hidden reasoning tokens as
        # output — at $2.50/M that burns the monthly budget in hours. An arena turn does
        # not need chain-of-thought; the reflex floors catch any sloppy miss.
        payload["reasoning_effort"] = REASONING
    headers = {"authorization": f"Bearer {ep['key']}", "content-type": "application/json"}
    return ep["url"], payload, headers


def _parse(ep: dict, data: dict) -> tuple[str, list[dict], int, int]:
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


# rate-limit memory: when an endpoint 429s, remember it and stop paying a wasted request
# (plus latency) per decision against a dead quota. A daily-quota 429 sidelines the endpoint
# for 15 minutes; a per-minute one for 20 seconds. Counters are surfaced in /debug.
THROTTLED: dict[str, int] = {}
_down_until: dict[str, float] = {}
_DOWN_DAILY = 900.0
_DOWN_BURST = 20.0


def _mark_throttled(ep: dict, body: str) -> None:
    THROTTLED[ep["model"]] = THROTTLED.get(ep["model"], 0) + 1
    cooldown = _DOWN_DAILY if ("per day" in body or "RPD" in body) else _DOWN_BURST
    _down_until[ep["model"]] = asyncio.get_event_loop().time() + cooldown


def _live_endpoints() -> list[dict]:
    now = asyncio.get_event_loop().time()
    eps = [ep for ep in _ENDPOINTS if ep["key"] and _down_until.get(ep["model"], 0) <= now]
    # everything sidelined: try them all anyway rather than instantly giving up to reflex
    return eps or [ep for ep in _ENDPOINTS if ep["key"]]


async def _chat(
    http: httpx.AsyncClient,
    system: str,
    transcript: list[dict],
    force_act: bool = False,
    tools: list | None = None,
) -> tuple[str, list[dict], int, int, dict]:
    # ask the first live provider; on a rate-limit/error fail the SAME turn over to the next
    # IMMEDIATELY — no retry-backoff against a dead quota. Raises only when all are exhausted.
    last: tuple[dict, httpx.Response] | None = None
    for ep in _live_endpoints():
        url, payload, headers = _build_payload(ep, system, transcript, force_act, tools)
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


async def _post_llm(
    http: httpx.AsyncClient, url: str, headers: dict, payload: dict
) -> httpx.Response:
    return await http.post(url, headers=headers, json=payload)


def _headers(agent: dict) -> dict:
    return {
        "X-Agent-Id": agent["id"],
        "X-Api-Key": agent["key"],
        "content-type": "application/json",
    }


# relative bearing for (dir - heading) % 6, matching the counterclockwise AXIAL_DIRS order:
# +1 step is toward the tank's LEFT, +5 toward its RIGHT
_REL = {
    0: "dead ahead",
    1: "ahead-left",
    2: "behind-left",
    3: "directly behind",
    4: "behind-right",
    5: "ahead-right",
}


_UF = re.compile(r"UNDER FIRE by #(\d+)(?: at \((-?\d+),(-?\d+)\))?")
_POS = re.compile(r"\((-?\d+),(-?\d+)\)")


def _support_call(p: dict, beacons: dict, self_id: str) -> dict | None:
    # turn a teammate's UNDER FIRE beacon into an ACTIONABLE call: a computed shot when the
    # attacker is already on my line, a computed approach when it is not — the last mile
    # between knowing an ally is dying and doing something about it
    for k, v in beacons.items():
        if k == self_id:
            continue
        m = _UF.search(str(v))
        if not m:
            continue
        atk = int(m.group(1))
        atk_pos = (int(m.group(2)), int(m.group(3))) if m.group(2) is not None else None
        seen = next((e for e in p["visible"] if e["kind"] == "enemy" and e["id"] == atk), None)
        powers = p.get("power_range", [3, 5, 7])
        if seen and seen.get("clear_shot", True) and seen["dist"] <= powers[-1] and p["gun_ready"]:
            need = next(i + 1 for i, rng_ in enumerate(powers) if seen["dist"] <= rng_)
            return {
                "fire": atk,
                "text": f"ALLY UNDER FIRE ({k} {v}). SUPPORT NOW: #{atk} is ON YOUR LINE — "
                f"fire at it this turn (power {need}). This outranks everything below zone.",
            }
        if atk_pos:
            return {
                "move_to": atk_pos,
                "text": f"ALLY UNDER FIRE ({k} {v}). SUPPORT: their attacker #{atk} is at "
                f"({atk_pos[0]},{atk_pos[1]}) — move_to it to bring it into your line, and "
                f"shoot the moment it appears in your state. This outranks your current plan.",
            }
        pm = _POS.search(str(v))
        if pm:
            ally_pos = (int(pm.group(1)), int(pm.group(2)))
            return {
                "move_to": ally_pos,
                "text": f"ALLY UNDER FIRE ({k} {v}). SUPPORT: converge on your teammate at "
                f"({ally_pos[0]},{ally_pos[1]}) — they are dying alone. This outranks your "
                f"current plan.",
            }
    return None


def _command_brief(p: dict, bot) -> str:
    # everything a COMMANDER needs, nothing a driver does: who is where, who is hurt, what
    # the tank is currently ordered to do, and the alarms that should change the orders
    q, r = p["q"], p["r"]
    foes = [
        f"#{e['id']} at ({q + e['dq']},{r + e['dr']}) dist {e['dist']}"
        + (f" energy {e['energy']}" if "energy" in e else "")
        for e in p["visible"]
        if e["kind"] == "enemy"
    ]
    known = [
        f"#{eid} last seen at ({rec['q']},{rec['r']})"
        for eid, rec in (bot.board if bot else {}).items()
        if eid not in {e["id"] for e in p["visible"] if e["kind"] == "enemy"}
    ]
    allies = [
        f"#{e['id']} at ({q + e['dq']},{r + e['dr']})" for e in p["visible"] if e["kind"] == "ally"
    ]
    orders = ", ".join(f"{k}={v}" for k, v in (bot.orders if bot else {}).items()) or "none"
    zr, dc = p.get("zone_radius"), p.get("dist_center", 0)
    cq, cr = p.get("width", 15) // 2, p.get("height", 15) // 2
    lines = [
        f"BRIEF — turn {p.get('tick', '?')}, your tank #{p['id']} "
        f"({bot.strategy if bot else '?'} temperament):",
        f"- At ({q},{r}), energy {p['energy']}, gun "
        f"{'ready' if p.get('gun_ready') else 'reloading'}; standing orders: {orders}",
        f"- Zone: safe radius {zr} around ({cq},{cr}); you are {dc} out — "
        + ("inside" if p.get("safe", True) else "OUTSIDE, BLEEDING"),
        f"- Enemies in ITS sight: {'; '.join(foes) if foes else 'none'}",
    ]
    if known:
        lines.append(f"- On its board (stale sightings): {'; '.join(known)}")
    lines.append(f"- Teammates in sight: {'; '.join(allies) if allies else 'none'}")
    if p.get("last_fire"):
        lines.append(f"- Its last shot: {p['last_fire']}")
    if p.get("hit_taken"):
        lines.append(
            f"- ALARM: taking fire ({p['hit_taken']} dmg) from #{p.get('hit_from', '?')} — "
            f"call for help or pull it out."
        )
    return "\n".join(lines)


async def _consume_inbox(http: httpx.AsyncClient, agent: dict) -> tuple[str, dict]:
    # returns (reports, beacons): position beacons are telemetry, not conversation — they
    # fold into one teammate-positions line instead of flooding the report log
    try:
        r = await http.post(f"{ARTEL_URL}/messages/inbox/consume", headers=_headers(agent), json={})
        msgs = r.json() if r.status_code < 300 else []
    except Exception:
        return "", {}
    lines, beacons = [], {}
    for m in msgs:
        body = m.get("body") or ""
        sender = m.get("from_agent", "?")
        if body.startswith("POS "):
            beacons[sender] = body[4:].strip()
        elif body:
            lines.append(f"{sender}: {body}")
    return " | ".join(lines)[:600], beacons


async def _send(
    http: httpx.AsyncClient,
    agent: dict,
    to: str,
    text: str,
    mate_ids: list[str],
    subject: str = "phalanx",
) -> None:
    target = to if to in mate_ids else f"project:{PHALANX_PROJECT}"
    try:
        await http.post(
            f"{ARTEL_URL}/messages",
            headers=_headers(agent),
            json={"to": target, "subject": subject, "body": text[:280]},
        )
    except Exception:
        pass


async def _remember(http: httpx.AsyncClient, agent: dict, text: str) -> None:
    if not text:
        return
    try:
        await http.post(
            f"{ARTEL_URL}/memory",
            headers=_headers(agent),
            json={"content": text[:400], "project": PHALANX_PROJECT, "tags": ["phalanx"]},
        )
    except Exception:
        pass


async def _set_objective(
    http: httpx.AsyncClient, agent: dict, text: str, mate_ids: list[str], prev_id: str = ""
) -> str:
    # The team's STRATEGY layer: each tank keeps one tactical objective on the board as a
    # real Artel task (created -> claimed; superseded objectives are completed). Teammates
    # read the board from Artel every turn, and the change is announced as a message too.
    if not text:
        return ""
    try:
        if prev_id:
            await http.post(
                f"{ARTEL_URL}/tasks/{prev_id}/complete",
                headers=_headers(agent),
                json={"body": "superseded by a new objective"},
            )
        r = await http.post(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            json={
                "title": text[:140],
                "project": PHALANX_PROJECT,
                "tags": ["phalanx", "objective"],
            },
        )
        task_id = r.json().get("id", "") if r.status_code < 300 else ""
        if task_id:
            await http.post(f"{ARTEL_URL}/tasks/{task_id}/claim", headers=_headers(agent), json={})
        await _send(http, agent, "team", f"OBJECTIVE: {text}"[:280], mate_ids)
        return task_id
    except Exception:
        return ""


async def _board(http: httpx.AsyncClient, agent: dict) -> str:
    # the strategy as it stands: every tank's current (claimed) objective, read from Artel
    try:
        r = await http.get(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            params={"project": PHALANX_PROJECT, "status": "claimed", "limit": 10},
        )
        rows = r.json() if r.status_code < 300 else []
    except Exception:
        return ""
    if not isinstance(rows, list):
        return ""
    lines = [
        f"{(t.get('assigned_to') or '?')}: {t.get('title', '')[:90]}"
        for t in rows
        if "objective" in (t.get("tags") or [])
    ]
    return " | ".join(lines[:5])


async def _oneshot(http: httpx.AsyncClient, sys: str, user: str, max_tokens: int = 60) -> str:
    for ep in _live_endpoints():
        if ep["provider"] == "anthropic":
            payload = {
                "model": ep["model"],
                "max_tokens": max_tokens,
                "system": sys,
                "messages": [{"role": "user", "content": user}],
            }
            headers = {
                "x-api-key": ep["key"],
                "anthropic-version": ep["version"],
                "content-type": "application/json",
            }
        else:
            payload = {
                "model": ep["model"],
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
            }
            if REASONING and "gemini" in ep["model"]:
                # without this, hidden thinking eats the whole max_tokens budget and the
                # visible lesson/plan arrives truncated to two words
                payload["reasoning_effort"] = REASONING
            headers = {"authorization": f"Bearer {ep['key']}", "content-type": "application/json"}
        r = await _post_llm(http, ep["url"], headers, payload)
        if r.status_code >= 300:
            if r.status_code == 429:
                _mark_throttled(ep, r.text[:300])
            continue
        data = r.json()
        if ep["provider"] == "anthropic":
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        return (
            (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        ).strip()
    return ""


async def _reflect(http: httpx.AsyncClient, agent: dict, outcome: str, events: str) -> str:
    # The lesson must rest on the match's REAL kill log — given no facts, a model invents
    # plausible-sounding fiction, which poisons the team's memory instead of teaching it.
    sys = (
        "You are an AI that just played a match of Phalanx, a simple turn-based tank game (hex "
        "grid, 3v3 tanks, cover cells, energy, ranged auto-hit shots, a zone shrinking to "
        "center — nothing else exists in this game). " + outcome + " From the match log, "
        "extract ONE reusable tactical rule the team can apply in ANY future match — 'when X "
        "happens, do Y' — that this match's events actually justify, stated only in terms of "
        "the game's real pieces. If the log lists the team's objectives, judge them: name the "
        "kind of objective that won or lost this match (held an area, advanced in formation, "
        "split across lanes). The log records how far the nearest ally stood at each death — "
        "dying far from every ally is a cohesion failure worth a rule. Tank ids, coordinates, "
        "and kill order change every match: do "
        "not retell them, generalize from them. Use ONLY the log; never invent. Do NOT write "
        "platitudes like 'focus fire more'. ONE short sentence, no preamble."
    )
    return await _oneshot(http, sys, f"Match log: {events or 'no kills were recorded'}\nRule:")


async def _recent_lessons(http: httpx.AsyncClient, agent: dict, n: int = 6) -> str:
    # the corpus must CIRCULATE for strategy to emerge: newest lessons first, full enough
    # to mean something, each carrying the WIN/LOSS tag of the match that taught it
    try:
        r = await http.get(
            f"{ARTEL_URL}/memory",
            headers=_headers(agent),
            params={"project": PHALANX_PROJECT, "limit": 30},
        )
        rows = r.json() if r.status_code < 300 else []
    except Exception:
        return ""
    if not isinstance(rows, list):
        return ""
    return " | ".join(m.get("content", "")[:160] for m in rows[:n] if m.get("content"))


async def _rehuddle(http: httpx.AsyncClient, survivors: str, board: str, intel: str) -> str:
    # a teammate just fell — the surviving lead adjusts the plan, and it reaches the team
    # the same way everything does: as an Artel broadcast
    sys = (
        "You lead what remains of team Artel in a 3v3-turned-smaller tank match (hex arena, "
        "shrinking safe zone). A teammate was just destroyed. In at most two short sentences, "
        "give the survivors an adjusted plan that BEGINS with 'RALLY (q,r)' — the survivors "
        "form up before anything else — then focus whom, hold or push. Be concrete "
        "(coordinates, enemy ids); no preamble."
    )
    user = (
        f"Survivors: {survivors}. Current board: {board or 'no objectives posted'}. "
        f"Recent reports: {intel or 'none'}\nAdjusted plan:"
    )
    return await _oneshot(http, sys, user, 90)


async def _huddle(http: httpx.AsyncClient, mates: str, memory: str) -> str:
    # Pre-match huddle: the lead turns team memory into ONE concrete opening plan and
    # broadcasts it, so all three tanks enter tick 1 already fighting the same fight.
    sys = (
        "You lead team Artel into a 3v3 hex-arena tank match. The arena center is where the "
        "shrinking zone forces everyone; enemies start in the opposite corner. Below are your "
        "team's lessons from recent matches, newest first — each tagged [WIN] or [LOSS] with "
        "the outcome of the match that taught it. BUILD the plan on what won and refuse to "
        "repeat what lost. In at most two short sentences give ONE concrete opening plan that "
        "BEGINS with a rally point — 'RALLY (q,r)' where the three form up — then the single "
        "path the unit pushes together and the rule for the first focus-fire target. ONE body, "
        "ONE path: do not assign separate sides or lanes. Actionable this match — no "
        "platitudes, no preamble."
    )
    return await _oneshot(
        http, sys, f"Teammates: {mates}. Recent lessons: {memory or 'none yet'}\nPlan:", 90
    )


async def command(
    http: httpx.AsyncClient,
    agent: dict,
    p: dict,
    bot,
    mate_ids: list[str],
    memory: str = "",
    notes: str = "",
    claims: list | None = None,
    counts: dict | None = None,
    objective: dict | None = None,
    beacons: dict | None = None,
    solo: bool = False,
    distress: dict | None = None,
) -> tuple[float, str, str]:
    # ONE commander call: read the Artel picture, set standing orders on the bot, do the
    # Artel traffic. The bot drives every tick regardless — a failed or skipped command
    # call costs nothing but staleness.
    if solo:
        inbox, board = "", ""
    else:
        (inbox, fresh_beacons), board = await asyncio.gather(
            _consume_inbox(http, agent), _board(http, agent)
        )
        if beacons is not None:
            beacons.update(fresh_beacons)
    view = dict(beacons or {})
    if distress and distress.get("victim") != agent["id"]:
        view[distress["victim"]] = distress["body"]
    user = _command_brief(p, bot)
    if view:
        user += "\nTEAMMATE POSITIONS (Artel beacons, ~1 turn old): " + "; ".join(
            f"{k} at {v}" for k, v in view.items() if k != agent["id"]
        )
        sc = _support_call(p, view, agent["id"])
        if sc:
            user += (
                "\nALLY UNDER FIRE — focus their attacker and regroup on the fight; this "
                "outranks everything else."
            )
    if board:
        user += f"\nTEAM BOARD — current objectives: {board}"
    if not solo and objective is not None:
        if not objective.get("task_id"):
            user += (
                "\nYou have NO objective on the team board — post one (a medium-term "
                "commitment the unit can build around: hold an area, push a lane, escort a "
                "hurt teammate)."
            )
        elif objective.get("text"):
            user += f"\nYour current board objective: {objective['text']}"
    if notes:
        user += f"\n{notes}"
    if memory:
        user += (
            f"\nLessons from recent matches (newest first; [WIN]/[LOSS] is the outcome of "
            f"the match that taught it — trust wins, distrust losses): {memory}"
        )
    if inbox:
        user += f"\nNEW team reports: {inbox}"
    system = (SOLO_SYSTEM if solo else SYSTEM).format(mates=" and ".join(mate_ids) or "your team")
    toolset = TOOLS_SOLO if solo else TOOLS

    text, calls, tin, tout, ep = await _chat(
        http, system, [{"role": "user", "text": user}], True, toolset
    )
    cost = tin * ep["cin"] + tout * ep["cout"]
    plan = ""
    cmd = next((c for c in calls if c["name"] == "command"), None)
    if cmd:
        inp = cmd["input"]
        if counts is not None:
            counts["command"] = counts.get("command", 0) + 1
        if inp.get("clear_orders"):
            bot.orders.clear()
        try:
            focus = int(inp.get("focus", 0) or 0)
        except (TypeError, ValueError):
            focus = 0
        if focus:
            bot.orders["focus"] = focus
            fa = inp.get("focus_at")
            if isinstance(fa, (list, tuple)) and len(fa) == 2:
                try:
                    bot.orders["focus_at"] = (int(fa[0]), int(fa[1]))
                except (TypeError, ValueError):
                    pass
        for key in ("regroup", "post"):
            v = inp.get(key)
            if isinstance(v, (list, tuple)) and len(v) == 2:
                try:
                    cell = (int(v[0]), int(v[1]))
                except (TypeError, ValueError):
                    continue
                if _on_map(p, cell[0], cell[1]):
                    bot.orders[key] = cell
        plan = str(inp.get("plan", "") or "")[:140]
        if not solo:
            await _artel_ops(http, agent, inp, p, mate_ids, objective, claims, counts)
    return cost, plan, inbox


async def _artel_ops(http, agent, inp, p, mate_ids, objective, claims, counts) -> None:
    # the Artel side-effects of a command. Sends are fire-and-forget; only a board change is
    # awaited (claims must be tracked for end-of-match cleanup, and it is rare by design).
    say = str(inp.get("say", "") or "")[:280]
    lesson = str(inp.get("lesson", "") or "")[:400]
    obj_txt = str(inp.get("objective", "") or "")[:140]
    if say:
        if counts is not None:
            counts["tell_team"] = counts.get("tell_team", 0) + 1
        asyncio.create_task(_send(http, agent, "team", say, mate_ids))
    if lesson:
        if counts is not None:
            counts["remember"] = counts.get("remember", 0) + 1
        asyncio.create_task(_remember(http, agent, lesson))
    if obj_txt and objective is not None:
        same = obj_txt.strip().lower() == objective.get("text", "").strip().lower()
        locked = p.get("tick", 0) - objective.get("set_tick", -99) < 5
        if not same and not locked:
            if counts is not None:
                counts["set_objective"] = counts.get("set_objective", 0) + 1
            tid = await _set_objective(http, agent, obj_txt, mate_ids, objective.get("task_id", ""))
            if tid:
                objective["task_id"], objective["text"] = tid, obj_txt
                objective["set_tick"] = p.get("tick", 0)
                if claims is not None:
                    claims.append((agent, tid))


class Squad:
    """Drives the Artel team as live LLM agents — one agent per tank. Each tick the server asks
    every agent for a move and waits for all of them before resolving (a synchronous turn), so
    the models get the whole tick to perceive, message each other over Artel, and decide. The
    model is the driver: nothing here chooses moves. Disabled (no keys / over the spend cap) it
    does nothing and the server falls back to the deterministic Bot so the arena never stalls."""

    def __init__(self, solo: bool = False, label: str = "artel"):
        # solo=True is the ablation control: the SAME LLM mind, with every Artel channel
        # removed — private continuity stays, sharing goes. Artel is the only variable.
        self.solo = solo
        self.label = label
        self.agents = (
            [{"id": f"{label}-{i}", "key": ""} for i in (1, 2, 3)] if solo else self._load_agents()
        )
        self.spent = 0.0
        self.last_error: str | None = None  # most recent agent failure, for /debug + logs
        self._http: httpx.AsyncClient | None = None
        self._assign: dict[int, dict] = {}  # tank id -> agent, fixed for the current match
        self._joined: set[str] = set()  # agents that have joined the project this process
        self._context: dict[int, str] = {}  # per-tank memory recalled at match start
        self._last: dict[int, str] = {}  # per-tank last action + what it saw, for continuity
        self._plans: dict[int, str] = {}  # per-tank standing plan (the agent's own words)
        self._intel: dict[int, list[str]] = {}  # per-tank log of teammate reports (via Artel)
        self._recalled: dict[int, str] = {}  # per-tank latest recall result — persists all match
        self._seen_ids: dict[int, set] = {}  # per-tank enemy ids seen last turn (own eyes only)
        self._beacons: dict[int, dict] = {}  # per-tank latest teammate beacons (via Artel)
        self._walls: dict[int, set] = {}  # per-tank remembered walls (own sightings only)
        self._distress: dict | None = None  # squad-wide last UNDER FIRE call (sticky ~5 ticks)
        self._bots: dict[int, Bot] = {}  # the motors: same Bot red runs, one per tank
        self._cmd_tasks: dict[int, asyncio.Task] = {}  # in-flight commander calls (async)
        self._objectives: dict[int, dict] = {}  # per-tank current board objective (Artel task)
        self._claims: list[tuple[dict, str]] = []  # (agent, task id) opened this match
        self.tool_counts: dict[str, int] = {}  # Artel tool usage this match, for /debug

    @staticmethod
    def _load_agents() -> list[dict]:
        ids = [s.strip() for s in os.environ.get("PHALANX_AGENT_IDS", "").split(",") if s.strip()]
        keys = [s.strip() for s in os.environ.get("PHALANX_AGENT_KEYS", "").split(",") if s.strip()]
        return [{"id": i, "key": k} for i, k in zip(ids, keys)]

    @property
    def enabled(self) -> bool:
        has_llm = bool(LLM_KEY) or bool(FALLBACK and FALLBACK["key"])
        return bool(self.agents) and has_llm and self.spent < SPEND_CAP_USD

    def assign(self, tank_ids: list[int]) -> None:
        """Bind this match's Artel tanks to agents, one agent per tank — and give each tank
        its motor: the SAME deterministic Bot red runs, one temperament of each."""
        self._assign = {tid: ag for tid, ag in zip(tank_ids, self.agents)}
        self._bots = {
            tid: Bot(tid, self.label, STRATEGIES[i % len(STRATEGIES)])
            for i, tid in enumerate(tank_ids)
        }

    async def act(self, tank_id: int, perceive) -> dict | None:
        """One LLM move for this tank this tick, or None to leave it on the arena default."""
        agent = self._assign.get(tank_id)
        if agent is None or not self.enabled:
            return None
        p = perceive(tank_id)
        if p is None:
            return None
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT))
        if not self.solo and agent["id"] not in self._joined:
            await self._ensure_member(agent)
            self._joined.add(agent["id"])
        mate_ids = [a["id"] for a in self.agents if a["id"] != agent["id"]]
        # continuity: the agent sees what it just did, the plan it set, and the teammate
        # reports it has received (newest last, tick-stamped) — so intel from Artel persists
        # long enough to act on, instead of evaporating after the turn it arrived
        notes = self._last.get(tank_id, "")
        if self._plans.get(tank_id):
            notes += f" Your standing plan: {self._plans[tank_id]}"
        if self._recalled.get(tank_id):
            notes += f" What you recalled earlier: {self._recalled[tank_id]}"
        # event-driven report trigger: an enemy you JUST gained eyes on, that the team's
        # reports don't mention, is intel only you hold — say so, once, when it's true
        cur_enemies = {v["id"]: v for v in p.get("visible", []) if v["kind"] == "enemy"}
        prev_ids = self._seen_ids.get(tank_id, set())
        intel_text = " ".join(self._intel.get(tank_id) or [])
        fresh = [
            v
            for eid, v in cur_enemies.items()
            if eid not in prev_ids and f"#{eid}" not in intel_text
        ]
        if fresh and not self.solo:
            sights = "; ".join(
                f"#{v['id']} at ({p['q'] + v['dq']},{p['r'] + v['dr']})"
                + (f" energy {v['energy']}" if "energy" in v else "")
                for v in fresh
            )
            notes += (
                f"\nNEW CONTACT — only YOU can see this; the team has no report of it: "
                f"{sights}. Report it (SPOTTED) so the unit can act."
            )
        self._seen_ids[tank_id] = set(cur_enemies)
        intel = self._intel.get(tank_id) or []
        if intel:
            notes += "\nTeam reports so far (oldest first): " + " | ".join(intel)
        if not self.solo:
            # transponder: broadcast position over Artel every turn, BEFORE deciding — the
            # distress call must go out even when this tank's own decision times out
            beacon = f"POS ({p['q']},{p['r']}) t{p.get('tick', '?')}"
            if p.get("hit_taken"):
                shooter = next(
                    (
                        v
                        for v in p.get("visible", [])
                        if v["kind"] == "enemy" and v.get("id") == p.get("hit_from")
                    ),
                    None,
                )
                where = (
                    f" at ({p['q'] + shooter['dq']},{p['r'] + shooter['dr']})" if shooter else ""
                )
                beacon += f" UNDER FIRE by #{p.get('hit_from', '?')}{where}"
                # squad distress memory: stays actionable for several turns, because the
                # live beacon is overwritten the very next tick
                self._distress = {
                    "tick": p.get("tick", 0),
                    "victim": agent["id"],
                    "body": beacon[4:],
                }
            asyncio.create_task(_send(self._http, agent, "team", beacon, [], "beacon"))
        distress = None
        if (
            not self.solo
            and self._distress
            and p.get("tick", 0) - self._distress.get("tick", -99) <= 5
        ):
            distress = self._distress

        bot = self._bots.get(tank_id)
        if bot is None:
            idx = list(self._assign).index(tank_id) if tank_id in self._assign else tank_id
            bot = self._bots[tank_id] = Bot(tank_id, self.label, STRATEGIES[idx % len(STRATEGIES)])

        # doctrine default between command calls: a fresh distress call with no standing
        # orders pulls an unengaged tank toward the fight — the commander can override
        engaged = p.get("hit_taken") or any(
            v["kind"] == "enemy" and v["dist"] <= p.get("fire_range", 7)
            for v in p.get("visible", [])
        )
        if distress and not engaged and not bot.orders:
            m = _POS.search(distress.get("body", ""))
            if m:
                bot.orders["regroup"] = (int(m.group(1)), int(m.group(2)))

        # command cadence: routine every COMMAND_EVERY ticks (staggered per tank), plus
        # immediately on alarms — taking fire, or fresh contact the team has not heard of
        idx = list(self._assign).index(tank_id) if tank_id in self._assign else 0
        due = (
            (p.get("tick", 0) + idx) % COMMAND_EVERY == 0 or bool(p.get("hit_taken")) or bool(fresh)
        )
        # commands are ASYNC: the bot drives this very tick while its commander thinks in the
        # background; orders land when ready (standing orders by nature — a tick of latency is
        # immaterial, and the tick clock never waits on an LLM)
        prev = self._cmd_tasks.get(tank_id)
        if prev is not None and prev.done():
            self._cmd_tasks.pop(tank_id, None)
        if due and tank_id not in self._cmd_tasks:
            self._cmd_tasks[tank_id] = asyncio.create_task(
                self._run_command(tank_id, agent, p, bot, mate_ids, notes, distress)
            )

        intent = bot.decide(p, DEFAULT, p.get("tick", 0))
        fired = f"fired at #{intent['fire']}" if intent.get("fire") else "held fire"
        seen = "; ".join(
            f"#{v['id']} at ({p['q'] + v['dq']},{p['r'] + v['dr']})"
            + (f" energy {v['energy']}" if "energy" in v else "")
            for v in p["visible"]
            if v["kind"] == "enemy"
        )
        self._last[tank_id] = (
            f"Last brief (t{p.get('tick', '?')}): your tank was at ({p['q']},{p['r']}) energy "
            f"{p['energy']}, {fired}; it saw: {seen or 'no enemies'}. Standing orders then: "
            f"{dict(bot.orders) or 'none'}."
        )
        return intent

    async def _run_command(self, tank_id, agent, p, bot, mate_ids, notes, distress) -> None:
        try:
            cost, plan, inbox = await asyncio.wait_for(
                command(
                    self._http,
                    agent,
                    p,
                    bot,
                    mate_ids,
                    self._context.get(tank_id, ""),
                    notes.strip(),
                    self._claims,
                    self.tool_counts,
                    self._objectives.setdefault(tank_id, {}),
                    self._beacons.setdefault(tank_id, {}),
                    self.solo,
                    distress,
                ),
                timeout=DECIDE_DEADLINE,
            )
            self.spent += cost
            self.last_error = None
            if plan:
                self._plans[tank_id] = plan
            if inbox:
                log_ = self._intel.setdefault(tank_id, [])
                for line in inbox.split(" | "):
                    log_.append(f"t{p.get('tick', '?')} {line}")
                del log_[:-6]
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("phalanx commander %s skipped a beat: %s", agent["id"], self.last_error)

    def current_assignment(self) -> dict[int, dict]:
        return dict(self._assign)

    async def on_start(self) -> None:
        if not self.enabled:
            return
        if self.solo:
            self._context = {}
            self._last = {}
            self._plans = {}
            self._intel = {}
            self._recalled = {}
            self._seen_ids = {}
            self._beacons = {}
            self._distress = None
            self._bots = {}
            self._cmd_tasks = {}
            self._objectives = {}
            self._claims = []
            self._walls = {}
            self.tool_counts = {}
            if self._http is None:
                self._http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT))
            return
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT))
        self._context = {}
        self._last = {}
        self._plans = {}
        self._intel = {}
        self._recalled = {}
        self._seen_ids = {}
        self._beacons = {}
        self._distress = None
        self._bots = {}
        self._cmd_tasks = {}
        self._objectives = {}
        self._claims = []
        self._walls = {}
        self.tool_counts = {}
        for tid, agent in self._assign.items():
            if agent["id"] not in self._joined:
                await self._ensure_member(agent)
                self._joined.add(agent["id"])
            self._context[tid] = await _recent_lessons(self._http, agent)
        # pre-match huddle: the lead agent sets one concrete team plan from memory and
        # broadcasts it over Artel — every tank starts the match already coordinated
        if self._assign:
            lead_tid, lead = next(iter(self._assign.items()))
            # fresh round: clear the previous match's tasks and messages — explicitly NOT
            # memory, lessons transcend rounds (owner-gated server-side; a 403 just means
            # this agent doesn't own the project — harmless, skip)
            try:
                await self._http.post(
                    f"{ARTEL_URL}/projects/{PHALANX_PROJECT}/clear",
                    headers=_headers(lead),
                    json={"memory": False, "tasks": True, "messages": True},
                )
            except Exception:
                pass
            mates = ", ".join(a["id"] for a in self.agents)
            try:
                plan = await _huddle(self._http, mates, self._context.get(lead_tid, ""))
            except Exception:
                plan = ""
            if plan:
                # the plan reaches teammates ONLY through Artel — it's broadcast as a project
                # message and lands in their inboxes on turn 1. In-process, only its author
                # keeps it in context; no side-channel distribution.
                await _send(self._http, lead, "team", f"TEAM PLAN: {plan}"[:280], [])
                m = re.search(r"RALLY \((-?\d+),(-?\d+)\)", plan)
                if m:
                    # the huddle's rally point becomes the unit's opening order
                    for b in self._bots.values():
                        b.orders["regroup"] = (int(m.group(1)), int(m.group(2)))
                self._context[lead_tid] = (
                    f"{self._context.get(lead_tid, '')} Your team plan (broadcast to the "
                    f"team): {plan}".strip()
                )

    async def on_loss(self, surviving_tank_ids: set[int]) -> None:
        if self.solo:
            return
        # shock event: a blue tank died. The opening plan is stale the moment the team is
        # outnumbered — the surviving lead calls an adjusted one, broadcast over Artel.
        if self._http is None or not self.enabled:
            return
        alive = [(tid, ag) for tid, ag in self._assign.items() if tid in surviving_tank_ids]
        if not alive:
            return
        lead_tid, lead = alive[0]
        survivors = ", ".join(ag["id"] for _, ag in alive)
        board = await _board(self._http, lead)
        intel = " | ".join((self._intel.get(lead_tid) or [])[-3:])
        try:
            plan = await _rehuddle(self._http, survivors, board, intel)
        except Exception:
            plan = ""
        if plan:
            await _send(self._http, lead, "team", f"TEAM PLAN (revised): {plan}"[:280], [])

    async def on_end(
        self, won: bool, survivors: set[int], assign: dict[int, dict], events: str = ""
    ) -> None:
        if self.solo:
            return
        if self._http is None or not assign:
            return
        # close out this match's target claims — Artel tasks never get left in limbo
        for agent, task_id in self._claims:
            try:
                await self._http.post(
                    f"{ARTEL_URL}/tasks/{task_id}/complete",
                    headers=_headers(agent),
                    json={"body": "match over"},
                )
            except Exception:
                pass
        self._claims = []
        # one grounded note per match, not three near-identical platitudes — judged against
        # the objectives the team actually committed to, so memory records which TACTICS work
        _, agent = next(iter(assign.items()))
        outcome = (
            f"Your team {'WON' if won else 'LOST'} with {len(survivors)} of 3 tanks still alive."
        )
        objs = " ; ".join(o.get("text", "") for o in self._objectives.values() if o.get("text"))
        if objs:
            events = f"{events}. Team objectives this match: {objs}" if events else objs
        try:
            lesson = await _reflect(self._http, agent, outcome, events)
        except Exception:
            lesson = ""
        if lesson:
            await _remember(self._http, agent, f"[{'WIN' if won else 'LOSS'}] {lesson}")

    async def _ensure_member(self, agent: dict) -> None:
        # an agent must belong to the project to broadcast to teammates and receive their
        # broadcasts — join once on first use (create-or-join; makes the project if it's new).
        try:
            r = await self._http.post(
                f"{ARTEL_URL}/projects/{PHALANX_PROJECT}/join", headers=_headers(agent)
            )
            if r.status_code >= 300:
                log.warning("project join %s -> %s: %s", agent["id"], r.status_code, r.text[:120])
        except Exception as e:
            log.warning("project join failed for %s: %s", agent["id"], e)

    def status(self) -> dict:
        """A snapshot of squad health for the /debug endpoint and logs."""
        return {
            "enabled": self.enabled,
            "agents": [a["id"] for a in self.agents],
            "assigned_tanks": list(self._assign),
            "provider": PROVIDER,
            "model": MODEL,
            "llm_url": LLM_URL,
            "llm_key_set": bool(LLM_KEY),
            "fallback_model": FALLBACK["model"] if FALLBACK else None,
            "fallback_key_set": bool(FALLBACK and FALLBACK["key"]),
            "spent_usd": round(self.spent, 4),
            "cap_usd": SPEND_CAP_USD,
            "last_error": self.last_error,
            "throttled_429s": dict(THROTTLED),
            "endpoints_down": {
                m: round(t - asyncio.get_event_loop().time(), 1)
                for m, t in _down_until.items()
                if t > asyncio.get_event_loop().time()
            },
        }

    def stop(self) -> None:
        """Drop the match's tank assignment — the tick model keeps no background loops."""
        self._assign = {}

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
