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
# Default is gpt-4o-mini over OpenAI's chat endpoint — cheap and reliable at tool-calling, which
# matters for actually firing. Set PHALANX_LLM_PROVIDER=anthropic (+ ANTHROPIC_API_KEY) for Claude,
# or point PHALANX_LLM_URL / PHALANX_MODEL / PHALANX_LLM_KEY at any other OpenAI-compatible provider.
PROVIDER = os.environ.get("PHALANX_LLM_PROVIDER", "openai")
MODEL = os.environ.get("PHALANX_MODEL", "gpt-4o-mini")
LLM_KEY = os.environ.get("PHALANX_LLM_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
_DEFAULT_URL = (
    "https://api.anthropic.com/v1/messages"
    if PROVIDER == "anthropic"
    else "https://api.openai.com/v1/chat/completions"
)
LLM_URL = os.environ.get("PHALANX_LLM_URL", _DEFAULT_URL)
LLM_VERSION = os.environ.get("PHALANX_LLM_VERSION", "2023-06-01")
SPEND_CAP_USD = float(os.environ.get("PHALANX_SPEND_CAP_USD", "20"))
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
    os.environ.get("PHALANX_COST_IN", "0.15"),
    os.environ.get("PHALANX_COST_OUT", "0.60"),
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
    "You command one tank on team Artel — your teammates {mates} share Artel with you — against "
    "three enemy tanks in a hex arena. Last team standing wins; a draw or losing your team is a "
    "loss, so play to WIN together. Each turn you take ONE action with the act tool.\n"
    "ARTEL is your team's coordination layer: the messages, lessons, and target claims you put "
    "into it steer real decisions by your teammates, this turn and next match. Everything you "
    "send through it must be about THIS arena, grounded in what you actually perceived — enemy "
    "ids, hex coordinates, energy, ticks, walls. Anything else pollutes the team's picture and "
    "loses games.\n"
    "TWO things must BOTH work every turn: (1) play YOUR tank well — fire when you can, take good "
    "ground, never just idle; AND (2) COORDINATE over Artel — share what you see, agree on a "
    "target, focus fire together. A team that only chats but plays sloppily loses; a tank that "
    "plays well but ignores its team loses. You need both at once.\n"
    "FIRING: your gun is target-based — set fire to an enemy id and it auto-hits any enemy IN "
    "RANGE with line of sight no matter which way you face (firing never needs turning). If the "
    "perception says you can fire, ALWAYS fire — you can move the same turn. A wasted gun loses. "
    "Saying you will fire is NOT firing: you must set fire to that enemy id in the act tool the "
    "SAME turn, or no shot happens.\n"
    "MOVING: never sit idle waiting. With no enemy in range, advance toward the arena center to "
    "make contact — the safe zone shrinks to the center, so camping the edge gets you killed. "
    "Turn toward where you want to go, then move fwd.\n"
    "STRATEGY — coordinate over Artel, do not fight solo: concentrate the team's fire on ONE "
    "enemy at a time (three guns destroy one tank fast, turning 3v3 into a 3v2 lead). USE Artel to "
    "make that happen — call the enemy you are firing on so the others pile onto it, act on what "
    "teammates tell you, and when you split up claim DIFFERENT enemies so no threat is left "
    "unwatched. Push toward the enemy together and stay close enough for crossfire; never wander "
    "off alone. Recall past lessons before you commit, and remember a CONCRETE one after a fight — "
    "but never let a message or a note cost you a shot."
)

TOOLS = [
    {
        "name": "tell_team",
        "description": "Send a SHORT call a teammate can act on THIS turn: the enemy you are "
        "firing on (so the team piles onto it), a threat closing on a teammate, or your "
        "position/energy when you are hurt and need cover. Read what teammates send you and act "
        "on it — this is not chatter, it is how three tanks fight as one.",
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
        "description": "Save ONE specific, concrete thing you actually observed this match that you "
        "could not have known in advance — a real enemy habit ('red rushes the center by tick 10'), "
        "a map spot that helped or got someone killed, or a move that clearly won or lost a fight. "
        "Do NOT save generic advice like 'focus fire the weakest' — that is already known and helps "
        "no one. It must be concrete enough to change a decision next match.",
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
        "different threats. Take one NO teammate has already claimed (check messages/recall first) "
        "— do not all claim the same id, that is just noise. Use it when the team should split, "
        "not when you are already focus-firing one enemy together.",
        "schema": {
            "type": "object",
            "properties": {"enemy_id": {"type": "integer"}},
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


async def _chat(
    http: httpx.AsyncClient, system: str, transcript: list[dict]
) -> tuple[str, list[dict], int, int, dict]:
    # ask the primary provider; on a rate-limit/error fail the SAME turn over to the fallback
    # provider so the model still drives. Raises only if every provider is exhausted.
    last: tuple[dict, httpx.Response] | None = None
    for ep in _ENDPOINTS:
        if not ep["key"]:
            continue
        url, payload, headers = _build_payload(ep, system, transcript)
        r = await _post_llm(http, url, headers, payload)
        if r.status_code < 300:
            text, calls, tin, tout = _parse(ep, r.json())
            return text, calls, tin, tout, ep
        log.warning("LLM %s HTTP %s: %s", ep["model"], r.status_code, r.text[:200])
        last = (ep, r)
    if last is None:
        raise RuntimeError("no llm endpoint configured")
    raise RuntimeError(f"llm http {last[1].status_code}")


async def _post_llm(
    http: httpx.AsyncClient, url: str, headers: dict, payload: dict
) -> httpx.Response:
    # OpenAI's lower tiers 429 under the burst of three agents calling at once each tick; a
    # short backoff (honoring Retry-After when given) usually clears it within the same tick,
    # which keeps the LLM driving instead of dropping to the deterministic fallback.
    r = None
    for attempt in range(3):
        r = await http.post(url, headers=headers, json=payload)
        if r.status_code != 429 and r.status_code < 500:
            return r
        if attempt == 2:
            return r
        ra = r.headers.get("retry-after", "")
        delay = float(ra) if ra.replace(".", "", 1).isdigit() else 0.6 * (attempt + 1)
        await asyncio.sleep(min(delay, 2.0))
    return r


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
    fr = p.get("fire_range", 6)
    foes, can_fire = [], []
    for e in p["visible"]:
        if e["kind"] != "enemy":
            continue
        rel = _REL[(e["dir"] - p["heading"]) % 6]
        in_range = e["dist"] <= fr
        if in_range:
            can_fire.append(str(e["id"]))
        energy = f" energy{e['energy']}" if "energy" in e else ""
        tag = "IN RANGE" if in_range else f"out of range (>{fr})"
        foes.append(f"#{e['id']} {rel} dist{e['dist']} [{tag}]{energy}")
    allies = [
        f"#{e['id']} {_REL[(e['dir'] - p['heading']) % 6]} dist{e['dist']}"
        for e in p["visible"]
        if e["kind"] == "ally"
    ]
    if not p["gun_ready"]:
        fire_line = "Your gun is RELOADING this turn. "
    elif can_fire:
        fire_line = f"You can FIRE this turn at: {', '.join(can_fire)} (no turning needed). "
    else:
        fire_line = "No enemy in range yet. "
    cdir = p.get("to_center")
    center_rel = _REL[(cdir - p["heading"]) % 6] if cdir is not None else None
    seen_foes = [e for e in p["visible"] if e["kind"] == "enemy"]
    move_line = ""
    if not can_fire and seen_foes:
        near = min(seen_foes, key=lambda e: e["dist"])
        rel = _REL[(near["dir"] - p["heading"]) % 6]
        move_line = (
            f"You SEE #{near['id']} {rel} at dist{near['dist']} but it's out of range — close in: "
            f"turn toward it and move fwd until it's in range, then fire. Don't hold still. "
        )
    elif not can_fire and center_rel and p.get("dist_center", 0) > 2:
        move_line = (
            f"No enemy in sight — the fight converges on the arena center as the zone shrinks. "
            f"Center is {center_rel} of you: turn toward it and move fwd to find the enemy. "
        )
    wallset = {(w["dq"], w["dr"]) for w in p.get("walls", [])}
    blocked = [_REL[(d - p["heading"]) % 6] for d in range(6) if AXIAL_DIRS[d] in wallset]
    cover_line = (
        f"COVER is right next to you — you CANNOT move into it, steer around it: {', '.join(blocked)}. "
        if blocked
        else ""
    )
    return (
        f"You are tank #{p['id']} at hex ({p['q']},{p['r']}), energy {p['energy']}, "
        f"{'inside' if p.get('safe', True) else 'OUTSIDE — taking zone damage'} the safe zone. "
        f"{fire_line}{move_line}{cover_line}"
        f"Enemies: {'; '.join(foes) if foes else 'none in sight'}. "
        f"Teammates: {', '.join(allies) if allies else 'none in sight'}."
    )


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


async def _claim_target(http: httpx.AsyncClient, agent: dict, enemy_id: int) -> None:
    if not enemy_id:
        return
    try:
        await http.post(
            f"{ARTEL_URL}/tasks",
            headers=_headers(agent),
            json={
                "title": f"destroy enemy {enemy_id}",
                "project": PHALANX_PROJECT,
                "assigned_to": agent["id"],
                "tags": ["phalanx", "target"],
            },
        )
    except Exception:
        pass


async def _reflect(http: httpx.AsyncClient, agent: dict, outcome: str) -> str:
    sys = (
        "You are an Artel tank writing a quick after-action note for your team. " + outcome + " In "
        "ONE short sentence, record the most SPECIFIC, concrete takeaway from THIS match worth "
        "remembering next time — a concrete enemy behaviour, a map spot that mattered, or a "
        "specific mistake to avoid. Be specific; do NOT write generic advice like 'focus fire'. "
        "No preamble."
    )
    for ep in _ENDPOINTS:
        if not ep["key"]:
            continue
        if ep["provider"] == "anthropic":
            payload = {
                "model": ep["model"],
                "max_tokens": 60,
                "system": sys,
                "messages": [{"role": "user", "content": "Lesson:"}],
            }
            headers = {
                "x-api-key": ep["key"],
                "anthropic-version": ep["version"],
                "content-type": "application/json",
            }
        else:
            payload = {
                "model": ep["model"],
                "max_tokens": 60,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": "Lesson:"},
                ],
            }
            headers = {"authorization": f"Bearer {ep['key']}", "content-type": "application/json"}
        r = await _post_llm(http, ep["url"], headers, payload)
        if r.status_code >= 300:
            continue
        data = r.json()
        if ep["provider"] == "anthropic":
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        return (
            (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        ).strip()
    return ""


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
    http: httpx.AsyncClient, agent: dict, p: dict, mate_ids: list[str], memory: str = ""
) -> tuple[dict, float]:
    inbox = await _consume_inbox(http, agent)
    user = _perception_text(p)
    if memory:
        user += f"\nTeam memory from past matches: {memory}"
    if inbox:
        user += f"\nFrom Artel teammates: {inbox}"
    system = SYSTEM.format(mates=" and ".join(mate_ids) or "your team")
    transcript: list[dict] = [{"role": "user", "text": user}]
    intent = {"turn": 0, "move": "hold", "fire": 0}
    cost = 0.0
    for _ in range(MAX_TOOL_ROUNDS):
        text, calls, tin, tout, ep = await _chat(http, system, transcript)
        cost += tin * ep["cin"] + tout * ep["cout"]
        if not calls:
            break
        transcript.append({"role": "assistant", "text": text, "calls": calls})
        results, acted = [], False
        for c in calls:
            if c["name"] == "act":
                inp = c["input"]
                intent = {
                    "turn": _TURN.get(inp.get("turn", "none"), 0),
                    "move": inp.get("move", "hold"),
                    "fire": int(inp.get("fire", 0) or 0),
                }
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
                await _claim_target(http, agent, int(c["input"].get("enemy_id", 0) or 0))
                results.append({"id": c["id"], "output": "claimed"})
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
    return intent, cost


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
        try:
            intent, cost = await asyncio.wait_for(
                decide(self._http, agent, p, mate_ids, self._context.get(tank_id, "")),
                timeout=DECIDE_DEADLINE,
            )
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("phalanx agent %s fell back to reflex: %s", agent["id"], self.last_error)
            return _reflex(p)
        self.spent += cost
        self.last_error = None
        return intent

    def current_assignment(self) -> dict[int, dict]:
        return dict(self._assign)

    async def on_start(self) -> None:
        if not self.enabled:
            return
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=httpx.Timeout(LLM_TIMEOUT))
        self._context = {}
        for tid, agent in self._assign.items():
            if agent["id"] not in self._joined:
                await self._ensure_member(agent)
                self._joined.add(agent["id"])
            self._context[tid] = await _recall(
                self._http,
                agent,
                "phalanx tactics: positioning, focus fire, the closing zone, beating the enemy team",
            )

    async def on_end(self, won: bool, survivors: set[int], assign: dict[int, dict]) -> None:
        if self._http is None or not assign:
            return
        # one grounded note per match, not three near-identical platitudes
        _, agent = next(iter(assign.items()))
        outcome = (
            f"Your team {'WON' if won else 'LOST'} with {len(survivors)} of 3 tanks still alive."
        )
        try:
            lesson = await _reflect(self._http, agent, outcome)
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
        }

    def stop(self) -> None:
        """Drop the match's tank assignment — the tick model keeps no background loops."""
        self._assign = {}

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
