from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

from .tank import bfs_step

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
MAX_TOOL_ROUNDS = 3


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
_GOAL_RULES = (
    "GOAL: destroy all three enemy tanks. Last team standing wins; a draw or mutual "
    "destruction is a loss.\n"
    "RULES:\n"
    "- Each turn all tanks act at once: one step of movement AND a shot, together.\n"
    "- MOVE BY DESTINATION: in act, set move_to [q,r] — any hex on the map. Your drivetrain "
    "takes the best path step toward it this turn (turning, routing around cover, queueing "
    "behind teammates). Think in destinations. Use move 'hold' only to deliberately stand "
    "still.\n"
    "- Energy is health and fuel; 0 = destroyed; no regeneration. A hit deals 12 and refunds "
    "the shooter 2.\n"
    "- Fire = enemy id + power. Power 1: range 3, FREE. Power 2: range 5, costs 2. Power 3: "
    "range 7, costs 4. No aiming, no missing: everything in your fire-now list is a "
    "guaranteed hit; anything else is a wasted trigger pull at full cost. The gun draws from "
    "YOUR energy — a shot that leaves you at 0 destroys you. Ready again next turn. Facing "
    "never matters for shooting.\n"
    "- A tank standing in the line of fire blocks the shot — friend or foe. Never fire "
    "through a teammate; against a stacked enemy, kill the front one first.\n"
    "- Cover blocks movement, shots, and sight. Tanks are solid: formation means adjacent "
    "hexes, never the same hex. Fog of war: you see distance 8 with a clear line; teammates "
    "see different things.\n"
    "- The safe zone shrinks toward the arena center; outside it you bleed energy every "
    "turn.\n"
)

SYSTEM = (
    "You are an AI playing Phalanx, a turn-based tank game on a hexagonal arena, driving one "
    "tank on team Artel against the three tanks of team Red. Your teammates are {mates}; the "
    "three of you share Artel and fight as a phalanx: one body, never three individuals.\n"
    + _GOAL_RULES
    + "PRIORITIES, in order — when they conflict, the higher one wins:\n"
    "1. ZONE: outside the safe zone, move toward the center every turn. Never hold there.\n"
    "2. FORMATION: stay within 2 hexes of a teammate. Their beacon positions are in your "
    "state every turn — out of formation means move_to a hex beside a teammate, now. Ahead "
    "of the unit? Wait for it.\n"
    "3. FIRE: a ready gun with a target in range always shoots, at the cheapest power that "
    "reaches. You can move the same turn.\n"
    "4. SUPPORT: a teammate UNDER FIRE (their beacon says so) outranks everything below — "
    "move to put a shot on their attacker, not toward the center. A unit that lets its "
    "tanks die alone loses one tank at a time.\n"
    "5. FOCUS: pour the unit's fire into ONE enemy — the one the team called. Never duel a "
    "kiter at long range; close to free-fire range behind cover, or refuse the fight.\n"
    "6. REPORT (Artel message): only what is NEW and actionable — 'SPOTTED #5 at (7,4) "
    "energy 40', 'TAKING FIRE at (6,8) from #4', 'FOCUS #5', 'REGROUP at (8,6)', 'AT (4,9) "
    "heading (7,6)' when nobody can see you. Nothing new = say nothing. Teammates' reports "
    "are real positions you cannot see — act on them.\n"
    "7. OBJECTIVE (Artel task board): your ONE medium-term commitment — 'hold around (8,6)', "
    "'push east with phalanx-blue-2'. It should survive 5+ turns; change it only when "
    "reality breaks it. The board in your state IS the team's strategy.\n"
    "MEMORY: lessons tagged [WIN]/[LOSS] from past matches arrive in your context — repeat "
    "what won, avoid what lost. Save one concrete, game-grounded lesson when you learn "
    "something real."
)

# the ablation control: the SAME mind with the Artel infrastructure removed. Identical
# game rules, identical ladder where possible; no radio, no board, no beacons, no memory.
SOLO_SYSTEM = (
    "You are an AI playing Phalanx, a turn-based tank game on a hexagonal arena, driving one "
    "tank on team Red against the three tanks of team Artel. Your teammates are {mates}, but "
    "you have NO communication with them — no radio, no shared map, no shared memory. You "
    "know only what you see.\n"
    + _GOAL_RULES
    + "PRIORITIES, in order — when they conflict, the higher one wins:\n"
    "1. ZONE: outside the safe zone, move toward the center every turn. Never hold there.\n"
    "2. FORMATION: stay within 2 hexes of a teammate YOU CAN SEE when possible — you have no "
    "other way to know where they are.\n"
    "3. FIRE: a ready gun with a target in range always shoots, at the cheapest power that "
    "reaches. You can move the same turn.\n"
    "4. FOCUS: concentrate your fire on one enemy at a time. Never duel a kiter at long "
    "range; close to free-fire range behind cover, or refuse the fight."
)

TOOLS = [
    {
        "name": "tell_team",
        "description": "Send a battlefield report (see REPORT priority): new, actionable, "
        "coordinate-bearing facts only. Duplicates and chatter are noise.",
        "schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "a teammate's agent id, or 'team' for both",
                },
                "text": {"type": "string"},
            },
            "required": ["to", "text"],
        },
    },
    {
        "name": "remember",
        "description": "Save ONE lesson that will still be true NEXT match, stated only in terms "
        "of the game's actual pieces (tanks, hexes, cover, energy, range, the zone) — a real "
        "enemy habit ('red rushes the center by tick 10'), or a move that clearly won or lost a "
        "fight. NOT current positions or energies (those die with this turn), NOT the team plan "
        "(that is a message), NOT generic advice like 'focus fire' (already known), and NOT "
        "anything involving mechanics this game does not have. If in doubt, don't save it.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": "Search your team's shared memory for concrete lessons from past matches "
        "(enemy habits, map hazards, what worked). Use it when you need to decide and prior "
        "experience would help — then act on what comes back.",
        "schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "set_objective",
        "description": "Post YOUR one medium-term objective on the team board (an Artel task): "
        "hold an area, push a lane with a teammate, destroy a target. It replaces your "
        "previous objective, is announced to the team, and should survive 5+ turns — read "
        "the board first and complement your teammates.",
        "schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "act",
        "description": "Set your tank's action for this turn. Ends your turn.",
        "schema": {
            "type": "object",
            "properties": {
                "move_to": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "hex [q,r] to step toward this turn — the drivetrain "
                    "handles turning and obstacles. OMIT move_to to stand still: holding "
                    "position while firing is fully respected and often correct.",
                },
                "move": {"type": "string", "enum": ["fwd", "back", "hold"]},
                "turn": {"type": "string", "enum": ["left", "right", "none"]},
                "fire": {
                    "type": "integer",
                    "description": "enemy tank id to shoot at, or 0 to hold fire",
                },
                "power": {
                    "type": "integer",
                    "description": "shot power 1-3 when firing: range 3/5/7 hexes for "
                    "0/2/4 energy (base shots are free); use the smallest power that reaches",
                },
                "plan": {
                    "type": "string",
                    "description": "optional: one short line on what you intend over the next "
                    "few turns (e.g. 'flanking right around the wall to reach #5') — shown back "
                    "to you next turn so you can follow through instead of starting over",
                },
            },
            "required": ["move", "turn", "fire"],
        },
    },
]

# AXIAL_DIRS rotates counterclockwise on screen (E, NE, NW, W, SW, SE) — so a LEFT turn
# (toward the tank's port side) is +1, and RIGHT is -1. These were mirrored for a long
# time, which made every steering decision come out backwards.
TOOLS_SOLO = [t for t in TOOLS if t["name"] == "act"]

_TURN = {"left": 1, "right": -1, "none": 0}


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
            "tool_choice": {"type": "tool", "name": "act"} if force_act else {"type": "any"},
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
        "tool_choice": {"type": "function", "function": {"name": "act"}}
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


def _perception_text(p: dict) -> str:
    # the agent's FULL state, every turn: itself (position, energy, cannon, gun), the zone,
    # everything it can see (enemies, teammates, cover) in absolute hex coordinates, and what
    # it can shoot right now. Decisions can only be as good as the state they rest on.
    fr = p.get("fire_range", 7)
    powers = p.get("power_range", [3, 5, 7])
    q, r, hd = p["q"], p["r"], p["heading"]
    foes, can_fire = [], []
    for e in p["visible"]:
        if e["kind"] != "enemy":
            continue
        rel = _REL[(e["dir"] - hd) % 6]
        clear = e.get("clear_shot", True)
        in_range = e["dist"] <= fr and clear
        need = next((i + 1 for i, rng_ in enumerate(powers) if e["dist"] <= rng_), 0)
        if in_range:
            can_fire.append(f"#{e['id']} (power {need})")
        energy = f", energy {e['energy']}" if "energy" in e else ""
        tag = (
            f"hit with power {need}+"
            if in_range
            else ("a tank blocks the shot" if not clear else f"out of range (>{fr})")
        )
        foes.append(
            f"#{e['id']} at ({q + e['dq']},{r + e['dr']}) dist {e['dist']} {rel} [{tag}]{energy}"
        )
    allies = [
        f"#{e['id']} at ({q + e['dq']},{r + e['dr']}) dist {e['dist']}"
        for e in p["visible"]
        if e["kind"] == "ally"
    ]
    walls = sorted((q + w["dq"], r + w["dr"]) for w in p.get("walls", []))
    fq, frr = AXIAL_DIRS[hd]
    cdir = p.get("to_center")
    center_rel = _REL[(cdir - hd) % 6] if cdir is not None else "toward mid-arena"
    zr = p.get("zone_radius")
    dc = p.get("dist_center", 0)

    w, h = p.get("width", 15), p.get("height", 15)
    wallset = {(w_["dq"], w_["dr"]) for w_ in p.get("walls", [])}
    occ = {(v["dq"], v["dr"]): v for v in p.get("visible", [])}
    blocked_dirs = []
    for d in range(6):
        dq, dr = AXIAL_DIRS[d]
        nq, nr = q + dq, r + dr
        if not _on_map(p, nq, nr):
            blocked_dirs.append(f"({nq},{nr}) [MAP EDGE]")
        elif (dq, dr) in wallset:
            blocked_dirs.append(f"({nq},{nr}) [cover]")
        elif (dq, dr) in occ:
            v = occ[(dq, dr)]
            who = "teammate" if v["kind"] == "ally" else "enemy"
            blocked_dirs.append(f"({nq},{nr}) [{who} #{v['id']} — tanks are solid]")
    lines = [
        f"STATE — turn {p.get('tick', '?')}, you are tank #{p['id']}:",
        f"- Arena: a HEXAGON of radius {(w - 1) // 2} around the center ({w // 2},{h // 2}) "
        f"(axial coords). Moving past its edge is impossible.",
        f"- Position ({q},{r}), energy {p['energy']}, cannon facing ({q + fq},{r + frr}) "
        f"[fwd moves there, back moves opposite; facing does not matter for shooting]",
        f"- Gun: {'READY' if p['gun_ready'] else 'reloading (ready next turn)'}"
        + (f"; you can fire NOW at: {', '.join(can_fire)}" if p["gun_ready"] and can_fire else ""),
        f"- Zone: safe radius {zr} around the arena center; you are {dc} from center — "
        + ("INSIDE the safe zone" if p.get("safe", True) else "OUTSIDE, BLEEDING energy"),
        f"- Enemies in sight: {'; '.join(foes) if foes else 'none'}",
        f"- Teammates in sight: {'; '.join(allies) if allies else 'none'}",
        "- Cover hexes in sight (impassable, block shots): "
        + (", ".join(f"({wq},{wr})" for wq, wr in walls) if walls else "none"),
    ]
    if blocked_dirs:
        lines.append(
            "- You CANNOT move into: " + ", ".join(blocked_dirs) + " — pick another direction."
        )
    costs = p.get("power_cost", [0, 2, 4])
    if p["energy"] <= costs[-1]:
        lines.append(
            f"- ENERGY CRITICAL ({p['energy']}): firing costs {costs[0]}/{costs[1]}/{costs[2]} "
            f"energy by power, and a shot that leaves you at 0 destroys you. Choose power you "
            f"survive — or spend yourself on a shot that matters."
        )
    if p.get("hit_taken"):
        shooter = p.get("hit_from", 0)
        seen_sh = next(
            (e for e in p["visible"] if e["kind"] == "enemy" and e["id"] == shooter), None
        )
        where = (
            f"#{shooter} at ({q + seen_sh['dq']},{r + seen_sh['dr']})"
            if seen_sh
            else f"#{shooter}, NOT in your sight (it shoots you from cover or beyond your vision)"
        )
        lines.append(
            f"- ALARM — YOU ARE TAKING FIRE: {p['hit_taken']} damage last turn from {where}. "
            f"Tell the team where, and either get help, break line of sight, or fall back to "
            f"a teammate. Standing alone under fire is how tanks die."
        )
    # urgency, where it changes the right move
    if not p.get("safe", True):
        lines.append(
            f"- ZONE: move toward the center ({center_rel} of you) NOW, before anything else."
        )
    elif zr is not None and zr - dc <= 2:
        lines.append(
            f"- ZONE WARNING: margin {round(zr - dc, 1)} — drift toward the center "
            f"({center_rel} of you) as you fight or the red cells swallow you."
        )
    if p["gun_ready"] and not can_fire:
        seen_foes = [e for e in p["visible"] if e["kind"] == "enemy"]
        if seen_foes:
            near = min(seen_foes, key=lambda e: e["dist"])
            rel = _REL[(near["dir"] - hd) % 6]
            lines.append(
                f"- You SEE #{near['id']} ({rel}, dist {near['dist']}) but it's out of range — "
                f"close in and fire; don't hold still."
            )
        elif p.get("dist_center", 0) > 2:
            lines.append(
                f"- No enemy in sight — the fight converges on the center ({center_rel} of "
                f"you) as the zone shrinks; advance to find them."
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
    return " | ".join(lines), beacons


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


async def _recall(http: httpx.AsyncClient, agent: dict, query: str) -> str:
    if not query:
        return ""
    try:
        r = await http.get(
            f"{ARTEL_URL}/memory/search",
            headers=_headers(agent),
            params={"q": query, "project": PHALANX_PROJECT, "limit": 5},
        )
        rows = r.json() if r.status_code < 300 else []
    except Exception:
        return ""
    if not isinstance(rows, list):
        return ""
    return " | ".join(m.get("content", "")[:200] for m in rows if m.get("content"))


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
        f"{(t.get('assigned_to') or '?')}: {t.get('title', '')}"
        for t in rows
        if "objective" in (t.get("tags") or [])
    ]
    return " | ".join(lines[:6])


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


async def _recent_lessons(http: httpx.AsyncClient, agent: dict, n: int = 8) -> str:
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
    return " | ".join(m.get("content", "")[:200] for m in rows[:n] if m.get("content"))


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


def _reflex(p: dict) -> dict:
    # a competent never-idle move computed from perception alone — used to fill gaps the model
    # leaves and as the fallback when a decision errors or runs past the deadline. Fire the
    # nearest in-range enemy; otherwise advance on the nearest enemy seen, else toward center.
    fr = p.get("fire_range", 7)
    powers = p.get("power_range", [3, 5, 7])
    seen = [v for v in p["visible"] if v["kind"] == "enemy"]
    in_range = [v for v in seen if v["dist"] <= fr and v.get("clear_shot", True)]
    costs = p.get("power_cost", [0, 2, 4])
    fire, power = 0, 0
    if p.get("gun_ready") and in_range:
        near = min(in_range, key=lambda v: v["dist"])
        need = next(i + 1 for i, rng_ in enumerate(powers) if near["dist"] <= rng_)
        if p.get("energy", 0) > costs[need - 1]:  # the backstop never spends the last point
            fire, power = near["id"], need
    tdir = None
    if seen:
        tdir = min(seen, key=lambda v: v["dist"])["dir"]
    elif p.get("dist_center", 0) > 1:
        tdir = p.get("to_center")
    if tdir is None:
        return {"turn": 0, "move": "hold", "fire": fire, "power": power}
    blocked = {(w["dq"], w["dr"]) for w in p.get("walls", [])}
    blocked |= {(v["dq"], v["dr"]) for v in p.get("visible", [])}  # tanks are solid
    for dd in range(6):
        dq, dr = AXIAL_DIRS[dd]
        if not _on_map(p, p["q"] + dq, p["r"] + dr):
            blocked.add((dq, dr))  # the map edge is as impassable as cover
    d = _open_dir(blocked, tdir)
    h = p["heading"]
    turn = 0 if h == d else (1 if (d - h) % 6 <= 3 else -1)
    return {"turn": turn, "move": "fwd", "fire": fire, "power": power}


async def decide(
    http: httpx.AsyncClient,
    agent: dict,
    p: dict,
    mate_ids: list[str],
    memory: str = "",
    notes: str = "",
    claims: list | None = None,
    counts: dict | None = None,
    objective: dict | None = None,
    beacons: dict | None = None,
    solo: bool = False,
    wall_mem: set | None = None,
) -> tuple[dict, float, str, str]:
    if solo:
        inbox, board = "", ""
    else:
        inbox, fresh_beacons = await _consume_inbox(http, agent)
        if beacons is not None:
            beacons.update(fresh_beacons)
        board = await _board(http, agent)
    user = _perception_text(p)
    if beacons:
        user += "\nTEAMMATE POSITIONS (Artel beacons, ~1 turn old): " + "; ".join(
            f"{k} at {v}" for k, v in beacons.items() if k != agent["id"]
        )
        under = [f"{k} {v}" for k, v in beacons.items() if k != agent["id"] and "UNDER FIRE" in v]
        if under and not p.get("hit_taken"):
            user += (
                "\nALLY UNDER FIRE — " + "; ".join(under) + ". Supporting them outranks "
                "your current plan: converge and put a shot on their attacker."
            )
    if board:
        user += f"\nTEAM BOARD — current objectives: {board}"
    if objective and not objective.get("task_id"):
        user += "\nYou have NO objective on the board yet — set one."
    if notes:
        user += f"\n{notes}"
    if memory:
        user += (
            f"\nLessons from recent matches (newest first; [WIN]/[LOSS] is the outcome of "
            f"the match that taught it — trust wins, distrust losses): {memory}"
        )
    if inbox:
        user += f"\nNEW team reports this turn: {inbox}"
    system = (SOLO_SYSTEM if solo else SYSTEM).format(mates=" and ".join(mate_ids) or "your team")
    toolset = TOOLS_SOLO if solo else None
    transcript: list[dict] = [{"role": "user", "text": user}]
    intent = {"turn": 0, "move": "hold", "fire": 0}
    cost, plan, recalled = 0.0, "", ""
    for round_i in range(MAX_TOOL_ROUNDS):
        # the LAST round forces the act tool: the model always chooses its own action;
        # the reflex is a failure backstop, not a co-pilot
        force_act = round_i == MAX_TOOL_ROUNDS - 1
        text, calls, tin, tout, ep = await _chat(http, system, transcript, force_act, toolset)
        cost += tin * ep["cin"] + tout * ep["cout"]
        if not calls:
            break
        transcript.append({"role": "assistant", "text": text, "calls": calls})
        results, acted = [], False
        for c in calls:
            if counts is not None:
                counts[c["name"]] = counts.get(c["name"], 0) + 1
            if c["name"] == "act":
                inp = c["input"]
                intent = {
                    "turn": _TURN.get(inp.get("turn", "none"), 0),
                    "move": inp.get("move", "hold"),
                    "fire": int(inp.get("fire", 0) or 0),
                    "power": int(inp.get("power", 0) or 0),
                }
                mt = inp.get("move_to")
                if isinstance(mt, (list, tuple)) and len(mt) == 2:
                    try:
                        intent["move_to"] = (int(mt[0]), int(mt[1]))
                    except (TypeError, ValueError):
                        pass
                plan = str(inp.get("plan", "") or "")[:140]
                acted = True
                results.append({"id": c["id"], "output": "ok"})
            elif c["name"] == "tell_team":
                await _send(
                    http,
                    agent,
                    str(c["input"].get("to", "team")),
                    str(c["input"].get("text", "")),
                    mate_ids,
                )
                results.append({"id": c["id"], "output": "sent"})
            elif c["name"] == "remember":
                await _remember(http, agent, str(c["input"].get("text", "")))
                results.append({"id": c["id"], "output": "saved"})
            elif c["name"] == "recall":
                found = await _recall(http, agent, str(c["input"].get("query", "")))
                recalled = found or recalled
                results.append({"id": c["id"], "output": found or "nothing relevant"})
            elif c["name"] == "set_objective":
                txt = str(c["input"].get("text", ""))[:140]
                if objective and txt.strip().lower() == objective.get("text", "").strip().lower():
                    results.append(
                        {"id": c["id"], "output": "already your objective — board unchanged"}
                    )
                    continue
                if objective and p.get("tick", 0) - objective.get("set_tick", -99) < 5:
                    results.append(
                        {
                            "id": c["id"],
                            "output": "objective locked (changes allowed every 5 turns) — "
                            "fight the plan you committed to",
                        }
                    )
                    continue
                tid = await _set_objective(
                    http, agent, txt, mate_ids, (objective or {}).get("task_id", "")
                )
                if tid:
                    if objective is not None:
                        objective["task_id"], objective["text"] = tid, txt
                        objective["set_tick"] = p.get("tick", 0)
                    if claims is not None:
                        claims.append((agent, tid))
                results.append({"id": c["id"], "output": "posted on the team board"})
        if acted:
            break
        transcript.append({"role": "tool", "results": results})
    # floors so a tank never under-acts: fire an in-range target the model ignored, and if it's
    # left holding with nothing to shoot, advance (on the nearest enemy, else toward center). The
    # model still drives whenever it makes a real choice; this only covers the turns it idles.
    rx = _reflex(p)
    # a shot the engine will reject (target out of range / behind cover / gun reloading) is a
    # PHANTOM: it costs nothing, hits nothing, and must not excuse holding position — tanks
    # have frozen for whole matches "firing" at an enemy through a wall
    powers = p.get("power_range", [3, 5, 7])
    dist_of = {
        v["id"]: v["dist"]
        for v in p["visible"]
        if v["kind"] == "enemy"
        and v["dist"] <= p.get("fire_range", 7)
        and v.get("clear_shot", True)
    }
    if intent.get("fire") and (intent["fire"] not in dist_of or not p.get("gun_ready")):
        intent["fire"] = 0
    if intent.get("fire"):
        need = next(i + 1 for i, rng_ in enumerate(powers) if dist_of[intent["fire"]] <= rng_)
        intent["power"] = max(int(intent.get("power", 0) or 0), need)  # never waste the pull
    if not intent.get("fire"):
        intent["fire"] = rx["fire"]
        if rx.get("power"):
            intent["power"] = rx["power"]
    if intent.get("move", "hold") == "hold" and not intent.get("fire"):
        intent["turn"], intent["move"] = rx["turn"], rx["move"]
    # drivetrain: the model thinks in destinations; we translate one optimal step. Choosing
    # WHERE to go is entirely the model's; turning mechanics are the tank's transmission.
    # Pathfinding is BFS over PERCEIVED walls (unknown terrain assumed open), so wall
    # pockets route around instead of trapping a greedy step; tanks only veto step one.
    mt = intent.pop("move_to", None)
    if mt and (mt[0], mt[1]) != (p["q"], p["r"]):
        wall_abs = {(p["q"] + w_["dq"], p["r"] + w_["dr"]) for w_ in p.get("walls", [])}
        if wall_mem is not None:
            # walls are static for the whole match: remember every one this tank has seen,
            # so cover discovered three turns ago still routes around when out of sight
            wall_mem |= wall_abs
            wall_abs = set(wall_mem)
        occ_abs = {(p["q"] + v["dq"], p["r"] + v["dr"]) for v in p.get("visible", [])}
        R_ = p.get("map_radius", (p.get("width", 15) - 1) // 2)
        best = bfs_step(p["q"], p["r"], mt[0], mt[1], wall_abs, R_, soft=occ_abs)
        if best is None:
            d0 = _hexdist(p["q"], p["r"], mt[0], mt[1])
            bd = None
            for dd in range(6):
                dq, dr = AXIAL_DIRS[dd]
                nq, nr = p["q"] + dq, p["r"] + dr
                if (nq, nr) in wall_abs or not _on_map(p, nq, nr):
                    continue
                prog = d0 - _hexdist(nq, nr, mt[0], mt[1])
                if bd is None or prog > bd:
                    best, bd = dd, prog
            if best is None:
                best = p["heading"]
        hh = p["heading"]
        intent["turn"] = 0 if hh == best else (1 if (best - hh) % 6 <= 3 else -1)
        travel = (hh + intent["turn"]) % 6
        intent["move"] = "fwd" if travel == best else "hold"  # rotate first if not facing yet
    # phantom-MOVE floor: a step into cover or off the map is silently rejected by the
    # engine — tanks have shoved a map edge for whole matches. Deflect toward the nearest
    # open direction (rotating first if the hull can't face it within one turn).
    if intent.get("move") in ("fwd", "back"):
        wallset = {(w_["dq"], w_["dr"]) for w_ in p.get("walls", [])}
        ally_rel = {(v["dq"], v["dr"]) for v in p.get("visible", []) if v["kind"] == "ally"}
        enemy_rel = {(v["dq"], v["dr"]) for v in p.get("visible", []) if v["kind"] == "enemy"}

        def _open(d: int) -> bool:
            dq, dr = AXIAL_DIRS[d]
            return (
                _on_map(p, p["q"] + dq, p["r"] + dr)
                and (dq, dr) not in wallset
                and (dq, dr) not in ally_rel
                and (dq, dr) not in enemy_rel
            )

        newh = (p["heading"] + intent.get("turn", 0)) % 6
        tdir = newh if intent["move"] == "fwd" else (newh + 3) % 6
        if not _open(tdir):
            dq, dr = AXIAL_DIRS[tdir]
            if (dq, dr) in ally_rel:
                # blocked by a TEAMMATE: queue behind it — hold this turn, keep facing.
                # Deflecting sideways here is what scattered the formation off the spawn.
                intent["move"] = "hold"
            else:
                for off in (1, -1, 2, -2, 3):
                    d2 = (tdir + off) % 6
                    if _open(d2):
                        hh = p["heading"]
                        intent["turn"] = 0 if hh == d2 else (1 if (d2 - hh) % 6 <= 3 else -1)
                        travel = (hh + intent["turn"]) % 6
                        intent["move"] = "fwd" if travel == d2 and _open(travel) else "hold"
                        break
    return intent, cost, plan, inbox, recalled


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
        """Bind this match's Artel tanks to agents, one agent per tank."""
        self._assign = {tid: ag for tid, ag in zip(tank_ids, self.agents)}

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
        try:
            intent, cost, plan, inbox, recalled = await asyncio.wait_for(
                decide(
                    self._http,
                    agent,
                    p,
                    mate_ids,
                    self._context.get(tank_id, ""),
                    notes.strip(),
                    self._claims,
                    self.tool_counts,
                    self._objectives.setdefault(tank_id, {}),
                    self._beacons.setdefault(tank_id, {}),
                    self.solo,
                    self._walls.setdefault(tank_id, set()),
                ),
                timeout=DECIDE_DEADLINE,
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("phalanx agent %s fell back to reflex: %s", agent["id"], self.last_error)
            return _reflex(p)
        self.spent += cost
        self.last_error = None
        if plan:
            self._plans[tank_id] = plan
        if recalled:
            self._recalled[tank_id] = recalled[:300]
        if not self.solo:
            # transponder: broadcast position over Artel every turn — equipment, not
            # strategy. Teammates read it from their inboxes.
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
            asyncio.create_task(_send(self._http, agent, "team", beacon, [], "beacon"))
        if inbox:
            log_ = self._intel.setdefault(tank_id, [])
            for line in inbox.split(" | "):
                log_.append(f"t{p.get('tick', '?')} {line}")
            del log_[:-6]
        fired = f"fired at #{intent['fire']}" if intent.get("fire") else "held fire"
        seen = "; ".join(
            f"#{v['id']} at ({p['q'] + v['dq']},{p['r'] + v['dr']})"
            + (f" energy {v['energy']}" if "energy" in v else "")
            for v in p["visible"]
            if v["kind"] == "enemy"
        )
        self._last[tank_id] = (
            f"Last turn (t{p.get('tick', '?')}) you were at ({p['q']},{p['r']}) with energy "
            f"{p['energy']}, {fired}, moved {intent.get('move', 'hold')}; you saw: "
            f"{seen or 'no enemies'}. If an enemy from then is missing from your state now, "
            f"it moved or broke line of sight — it still exists."
        )
        return intent

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
