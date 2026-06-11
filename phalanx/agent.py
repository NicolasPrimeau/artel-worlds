from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx

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
MODEL = os.environ.get("PHALANX_MODEL", "gemini-2.5-flash")
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
    os.environ.get("PHALANX_COST_IN", "0.30"),
    os.environ.get("PHALANX_COST_OUT", "2.50"),
)
_LLM2_KEY = os.environ.get("PHALANX_LLM2_KEY", "")
FALLBACK = (
    _make_ep(
        os.environ.get("PHALANX_LLM2_PROVIDER", "openai"),
        os.environ.get("PHALANX_LLM2_MODEL", "gemini-2.5-flash-lite"),
        os.environ.get(
            "PHALANX_LLM2_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        ),
        _LLM2_KEY,
        os.environ.get("PHALANX_LLM2_VERSION", "2023-06-01"),
        os.environ.get("PHALANX_LLM2_COST_IN", "0.10"),
        os.environ.get("PHALANX_LLM2_COST_OUT", "0.40"),
    )
    if _LLM2_KEY
    else None
)
_ENDPOINTS = [PRIMARY] + ([FALLBACK] if FALLBACK else [])
# Hard ceiling on a tank's WHOLE decision (all tool rounds together) so a slow model can never
# stall the synchronous tick — past it the tank falls back to a reflex move instead of freezing.
DECIDE_DEADLINE = float(os.environ.get("PHALANX_DECIDE_DEADLINE", "8"))
LLM_TIMEOUT = float(os.environ.get("PHALANX_LLM_TIMEOUT", "12"))

# Concise. Names the teammates, says coordinate through Artel — and stops there. No
# instructions on HOW to use Artel: the coordination has to emerge, not be scripted.
SYSTEM = (
    "You are an AI playing Phalanx, a turn-based tank game on a bounded hex grid, driving one "
    "tank on team Artel — your teammates {mates} share Artel with you — against the three "
    "tanks of team Red.\n"
    "GOAL: destroy all three enemy tanks. Last team standing wins the match; a draw or mutual "
    "destruction counts as a loss, so play to WIN together.\n"
    "RULES OF THE GAME:\n"
    "- All tanks act simultaneously each turn. In one turn you may turn (left/right, one of 6 "
    "facings), move one hex (fwd/back along your facing), AND fire — all together.\n"
    "- Energy is health and fuel in one number; at 0 your tank is destroyed; it never "
    "regenerates. A hit costs the target 12 energy; landing a hit refunds the shooter 2.\n"
    "- Firing is target-based: name an enemy id and the shot automatically hits if that enemy "
    "is within range 6 with a clear line — your facing does NOT matter for shooting, only for "
    "moving. The gun is ready again the very next turn.\n"
    "- Cover hexes are impassable and block both shots and sight. You see only what has a "
    "clear line to you within distance 8 — fog of war; your teammates see different things.\n"
    "- The safe zone shrinks toward the arena center as the match goes on. Any tank outside it "
    "loses energy every turn: when you are outside, holding position is forbidden — move "
    "toward the center every turn until you are safe (you can still fire while moving).\n"
    "ARTEL is your team's coordination layer: the messages, lessons, and target claims you put "
    "into it steer real decisions by your teammates, this turn and next match. Everything you "
    "send through it must be about THIS arena, grounded in what you actually perceived — enemy "
    "ids, hex coordinates, energy, ticks, walls.\n"
    "TWO things must BOTH work every turn: (1) play YOUR tank well — fire whenever you can "
    "(a ready gun with a target in range should always shoot; you can move the same turn), "
    "take good ground, never idle; AND (2) COORDINATE over Artel — share what you see, agree "
    "on a target, focus fire together. A team that only chats but plays sloppily loses; a tank "
    "that plays well but ignores its team loses.\n"
    "ARRIVE TOGETHER: while advancing stay within 2 hexes of a teammate, and if you are ahead "
    "of your team, hold or angle back until they catch up — the tank that reaches the enemy "
    "first fights 3-vs-1 and dies before the team can trade back.\n"
    "STRATEGY — the team fights ONE fight, not three: there is a TEAM PLAN (from the pre-match "
    "huddle, updated over Artel as the fight turns) and following it beats any solo brilliance. "
    "Concentrate the team's fire on ONE enemy at a time (three guns destroy one tank fast, "
    "turning 3v3 into a 3v2 lead). USE Artel to make that happen — call the enemy you are firing "
    "on so the others pile onto it, act on what teammates tell you, and if the plan stops "
    "working say so and call the adjustment. Push toward the enemy together and stay close "
    "enough for crossfire; never wander off alone. Recall past lessons before you commit, and "
    "remember a CONCRETE one after a fight — but never let a message or a note cost you a shot."
)

TOOLS = [
    {
        "name": "tell_team",
        "description": "Send REAL INTEL a teammate can act on: enemy id + hex coordinates + "
        "energy ('enemy #5 at (8,4) energy 23 — firing on it, pile on'), a threat's position, or "
        "your own position/energy when hurt. Teammates CANNOT see what you see — fog of war — so "
        "a sighting you don't share with its coordinates is intel the team doesn't have. Vague "
        "chatter ('be careful', 'let's win') is noise; coordinates are coordination.",
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
        "name": "claim_target",
        "description": "Claim an enemy id as YOUR target so the team DIVIDES enemies and covers "
        "different threats. Include where you last saw it so teammates know which sector is "
        "covered. Take one NO teammate has already claimed (check messages/recall first) — do "
        "not all claim the same id, that is just noise. Use it when the team should split, not "
        "when you are already focus-firing one enemy together.",
        "schema": {
            "type": "object",
            "properties": {
                "enemy_id": {"type": "integer"},
                "q": {"type": "integer", "description": "hex q where you last saw it"},
                "r": {"type": "integer", "description": "hex r where you last saw it"},
            },
            "required": ["enemy_id"],
        },
    },
    {
        "name": "act",
        "description": "Set your tank's action for this turn. Ends your turn.",
        "schema": {
            "type": "object",
            "properties": {
                "move": {"type": "string", "enum": ["fwd", "back", "hold"]},
                "turn": {"type": "string", "enum": ["left", "right", "none"]},
                "fire": {
                    "type": "integer",
                    "description": "enemy tank id to shoot at, or 0 to hold fire",
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

_TURN = {"left": -1, "right": 1, "none": 0}
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
def _build_payload(ep: dict, system: str, transcript: list[dict]) -> tuple[str, dict, dict]:
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
            for t in TOOLS
        ],
        "tool_choice": "required",
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
    http: httpx.AsyncClient, system: str, transcript: list[dict]
) -> tuple[str, list[dict], int, int, dict]:
    # ask the first live provider; on a rate-limit/error fail the SAME turn over to the next
    # IMMEDIATELY — no retry-backoff against a dead quota. Raises only when all are exhausted.
    last: tuple[dict, httpx.Response] | None = None
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


_REL = {
    0: "dead ahead",
    1: "ahead-right",
    2: "behind-right",
    3: "directly behind",
    4: "behind-left",
    5: "ahead-left",
}


def _perception_text(p: dict) -> str:
    # the agent's FULL state, every turn: itself (position, energy, cannon, gun), the zone,
    # everything it can see (enemies, teammates, cover) in absolute hex coordinates, and what
    # it can shoot right now. Decisions can only be as good as the state they rest on.
    fr = p.get("fire_range", 6)
    q, r, hd = p["q"], p["r"], p["heading"]
    foes, can_fire = [], []
    for e in p["visible"]:
        if e["kind"] != "enemy":
            continue
        rel = _REL[(e["dir"] - hd) % 6]
        in_range = e["dist"] <= fr
        if in_range:
            can_fire.append(str(e["id"]))
        energy = f", energy {e['energy']}" if "energy" in e else ""
        tag = "IN RANGE" if in_range else f"out of range (>{fr})"
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

    lines = [
        f"STATE — turn {p.get('tick', '?')}, you are tank #{p['id']}:",
        f"- Position ({q},{r}), energy {p['energy']}, cannon facing ({q + fq},{r + frr}) "
        f"[fwd moves there, back moves opposite; facing does not matter for shooting]",
        f"- Gun: {'READY' if p['gun_ready'] else 'reloading (ready next turn)'}"
        + (f"; you can fire NOW at: {', '.join(can_fire)}" if p["gun_ready"] and can_fire else ""),
        f"- Zone: safe radius {zr} around the arena center; you are {dc} from center — "
        + ("INSIDE the safe zone" if p.get("safe", True) else "OUTSIDE, BLEEDING energy"),
        f"- Enemies in sight: {'; '.join(foes) if foes else 'none'}",
        f"- Teammates in sight: {'; '.join(allies) if allies else 'none'}",
        f"- Cover hexes in sight (impassable, block shots): "
        + (", ".join(f"({wq},{wr})" for wq, wr in walls) if walls else "none"),
    ]
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


async def _consume_inbox(http: httpx.AsyncClient, agent: dict) -> str:
    try:
        r = await http.post(f"{ARTEL_URL}/messages/inbox/consume", headers=_headers(agent), json={})
        msgs = r.json() if r.status_code < 300 else []
    except Exception:
        return ""
    lines = [f"{m.get('from_agent', '?')}: {m.get('body', '')}" for m in msgs if m.get("body")]
    return " | ".join(lines)


async def _send(
    http: httpx.AsyncClient, agent: dict, to: str, text: str, mate_ids: list[str]
) -> None:
    target = to if to in mate_ids else f"project:{PHALANX_PROJECT}"
    try:
        await http.post(
            f"{ARTEL_URL}/messages",
            headers=_headers(agent),
            json={"to": target, "subject": "phalanx", "body": text[:280]},
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
            params={"q": query, "project": PHALANX_PROJECT, "limit": 3},
        )
        rows = r.json() if r.status_code < 300 else []
    except Exception:
        return ""
    if not isinstance(rows, list):
        return ""
    return " | ".join(m.get("content", "")[:140] for m in rows if m.get("content"))


async def _claim_target(
    http: httpx.AsyncClient, agent: dict, enemy_id: int, mate_ids: list[str], at: str = ""
) -> str:
    # A claim only coordinates if teammates SEE it — broadcast it to the project (it lands in
    # their inbox next turn) WITH the target's last-seen position, so the team knows which
    # sector is covered. The Artel task makes the claim visible/auditable in the UI and is
    # claimed properly so the squad can complete it at match end instead of leaving it open.
    if not enemy_id:
        return ""
    where = f" last seen at {at}" if at else ""
    await _send(
        http, agent, "team", f"claiming enemy #{enemy_id}{where} — it's mine to cover", mate_ids
    )
    try:
        r = await http.post(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            json={
                "title": f"destroy enemy {enemy_id}{where}",
                "project": PHALANX_PROJECT,
                "tags": ["phalanx", "target"],
            },
        )
        task_id = r.json().get("id", "") if r.status_code < 300 else ""
        if task_id:
            await http.post(f"{ARTEL_URL}/tasks/{task_id}/claim", headers=_headers(agent), json={})
        return task_id
    except Exception:
        return ""


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
        "the game's real pieces. Tank ids, coordinates, and kill order change every match: do "
        "not retell them, generalize from them (e.g. losses to the closing zone -> a rule about "
        "when to rotate in; first kill won the fight -> a rule about forcing an early pick). "
        "Use ONLY the log; never invent. Do NOT write platitudes like 'focus fire more'. ONE "
        "short sentence, no preamble."
    )
    return await _oneshot(http, sys, f"Match log: {events or 'no kills were recorded'}\nRule:")


async def _huddle(http: httpx.AsyncClient, mates: str, memory: str) -> str:
    # Pre-match huddle: the lead turns team memory into ONE concrete opening plan and
    # broadcasts it, so all three tanks enter tick 1 already fighting the same fight.
    sys = (
        "You lead team Artel into a 3v3 hex-arena tank match. The arena center is where the "
        "shrinking zone forces everyone; enemies start in the opposite corner. In at most two "
        "short sentences give your team ONE concrete opening plan: where you three push together, "
        "the rule for choosing the first focus-fire target, and who covers which side. Make it "
        "actionable this match — no platitudes, no preamble."
    )
    return await _oneshot(
        http, sys, f"Teammates: {mates}. Team memory: {memory or 'none yet'}\nPlan:", 90
    )


def _reflex(p: dict) -> dict:
    # a competent never-idle move computed from perception alone — used to fill gaps the model
    # leaves and as the fallback when a decision errors or runs past the deadline. Fire the
    # nearest in-range enemy; otherwise advance on the nearest enemy seen, else toward center.
    fr = p.get("fire_range", 6)
    seen = [v for v in p["visible"] if v["kind"] == "enemy"]
    in_range = [v for v in seen if v["dist"] <= fr]
    fire = min(in_range, key=lambda v: v["dist"])["id"] if (p.get("gun_ready") and in_range) else 0
    tdir = None
    if seen:
        tdir = min(seen, key=lambda v: v["dist"])["dir"]
    elif p.get("dist_center", 0) > 1:
        tdir = p.get("to_center")
    if tdir is None:
        return {"turn": 0, "move": "hold", "fire": fire}
    wallset = {(w["dq"], w["dr"]) for w in p.get("walls", [])}
    d = _open_dir(wallset, tdir)
    h = p["heading"]
    turn = 0 if h == d else (1 if (d - h) % 6 <= 3 else -1)
    return {"turn": turn, "move": "fwd", "fire": fire}


async def decide(
    http: httpx.AsyncClient,
    agent: dict,
    p: dict,
    mate_ids: list[str],
    memory: str = "",
    notes: str = "",
    claims: list | None = None,
    counts: dict | None = None,
) -> tuple[dict, float, str]:
    inbox = await _consume_inbox(http, agent)
    user = _perception_text(p)
    if notes:
        user += f"\n{notes}"
    if memory:
        user += f"\nTeam memory from past matches: {memory}"
    if inbox:
        user += f"\nFrom Artel teammates: {inbox}"
    system = SYSTEM.format(mates=" and ".join(mate_ids) or "your team")
    transcript: list[dict] = [{"role": "user", "text": user}]
    intent = {"turn": 0, "move": "hold", "fire": 0}
    cost, plan = 0.0, ""
    for _ in range(MAX_TOOL_ROUNDS):
        text, calls, tin, tout, ep = await _chat(http, system, transcript)
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
                }
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
                results.append({"id": c["id"], "output": found or "nothing relevant"})
            elif c["name"] == "claim_target":
                inp = c["input"]
                at = (
                    f"({inp.get('q')},{inp.get('r')})"
                    if inp.get("q") is not None and inp.get("r") is not None
                    else ""
                )
                tid = await _claim_target(
                    http, agent, int(inp.get("enemy_id", 0) or 0), mate_ids, at
                )
                if tid and claims is not None:
                    claims.append((agent, tid))
                results.append({"id": c["id"], "output": "claimed and announced to the team"})
        if acted:
            break
        transcript.append({"role": "tool", "results": results})
    # floors so a tank never under-acts: fire an in-range target the model ignored, and if it's
    # left holding with nothing to shoot, advance (on the nearest enemy, else toward center). The
    # model still drives whenever it makes a real choice; this only covers the turns it idles.
    rx = _reflex(p)
    if not intent.get("fire"):
        intent["fire"] = rx["fire"]
    if intent.get("move", "hold") == "hold" and not intent.get("fire"):
        intent["turn"], intent["move"] = rx["turn"], rx["move"]
    return intent, cost, plan


class Squad:
    """Drives the Artel team as live LLM agents — one agent per tank. Each tick the server asks
    every agent for a move and waits for all of them before resolving (a synchronous turn), so
    the models get the whole tick to perceive, message each other over Artel, and decide. The
    model is the driver: nothing here chooses moves. Disabled (no keys / over the spend cap) it
    does nothing and the server falls back to the deterministic Bot so the arena never stalls."""

    def __init__(self):
        self.agents = self._load_agents()
        self.spent = 0.0
        self.last_error: str | None = None  # most recent agent failure, for /debug + logs
        self._http: httpx.AsyncClient | None = None
        self._assign: dict[int, dict] = {}  # tank id -> agent, fixed for the current match
        self._joined: set[str] = set()  # agents that have joined the project this process
        self._context: dict[int, str] = {}  # per-tank memory recalled at match start
        self._last: dict[int, str] = {}  # per-tank last action, carried into the next turn
        self._plans: dict[int, str] = {}  # per-tank standing plan (the agent's own words)
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
        if agent["id"] not in self._joined:
            await self._ensure_member(agent)
            self._joined.add(agent["id"])
        mate_ids = [a["id"] for a in self.agents if a["id"] != agent["id"]]
        # continuity: the agent sees what it just did and the plan it set, so it can follow
        # through across turns instead of rediscovering the situation from scratch every 2.5s
        notes = self._last.get(tank_id, "")
        if self._plans.get(tank_id):
            notes += f" Your standing plan: {self._plans[tank_id]}"
        try:
            intent, cost, plan = await asyncio.wait_for(
                decide(
                    self._http,
                    agent,
                    p,
                    mate_ids,
                    self._context.get(tank_id, ""),
                    notes.strip(),
                    self._claims,
                    self.tool_counts,
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
        fired = f"fired at #{intent['fire']}" if intent.get("fire") else "held fire"
        self._last[tank_id] = f"Last turn you {fired} and moved {intent.get('move', 'hold')}."
        return intent

    def current_assignment(self) -> dict[int, dict]:
        return dict(self._assign)

    async def on_start(self) -> None:
        if not self.enabled:
            return
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT))
        self._context = {}
        self._last = {}
        self._plans = {}
        self._claims = []
        self.tool_counts = {}
        for tid, agent in self._assign.items():
            if agent["id"] not in self._joined:
                await self._ensure_member(agent)
                self._joined.add(agent["id"])
            self._context[tid] = await _recall(
                self._http,
                agent,
                "phalanx tactics: positioning, focus fire, the closing zone, beating the enemy team",
            )
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
                await _send(self._http, lead, "team", f"TEAM PLAN: {plan}"[:280], [])
                for tid in self._assign:
                    self._context[tid] = f"{self._context.get(tid, '')} Team plan: {plan}".strip()

    async def on_end(
        self, won: bool, survivors: set[int], assign: dict[int, dict], events: str = ""
    ) -> None:
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
        # one grounded note per match, not three near-identical platitudes
        _, agent = next(iter(assign.items()))
        outcome = (
            f"Your team {'WON' if won else 'LOST'} with {len(survivors)} of 3 tanks still alive."
        )
        try:
            lesson = await _reflect(self._http, agent, outcome, events)
        except Exception:
            lesson = ""
        if lesson:
            await _remember(self._http, agent, lesson)

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
