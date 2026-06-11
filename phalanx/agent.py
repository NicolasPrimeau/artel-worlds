from __future__ import annotations

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
# Default is Gemini Flash over its OpenAI-compatible endpoint — ~10x cheaper than Haiku.
# Set PHALANX_LLM_PROVIDER=anthropic (+ ANTHROPIC_API_KEY) to swap to Claude, or point
# PHALANX_LLM_URL / PHALANX_MODEL at any other OpenAI-compatible provider.
PROVIDER = os.environ.get("PHALANX_LLM_PROVIDER", "openai")
MODEL = os.environ.get("PHALANX_MODEL", "gemini-2.5-flash-lite")
LLM_KEY = os.environ.get("PHALANX_LLM_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
_DEFAULT_URL = (
    "https://api.anthropic.com/v1/messages"
    if PROVIDER == "anthropic"
    else "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
)
LLM_URL = os.environ.get("PHALANX_LLM_URL", _DEFAULT_URL)
LLM_VERSION = os.environ.get("PHALANX_LLM_VERSION", "2023-06-01")
SPEND_CAP_USD = float(os.environ.get("PHALANX_SPEND_CAP_USD", "20"))
COST_IN = float(os.environ.get("PHALANX_COST_IN", "0.10")) / 1_000_000  # $/input token (Flash)
COST_OUT = float(os.environ.get("PHALANX_COST_OUT", "0.40")) / 1_000_000  # $/output token (Flash)
MAX_TOOL_ROUNDS = 4
# Per-tick decision deadline. The synchronous tick waits for every agent's move before it
# resolves, so this caps how long one model may take; past it that tank holds for the tick.
LLM_TIMEOUT = float(os.environ.get("PHALANX_LLM_TIMEOUT", "15"))

# Concise. Names the teammates, says coordinate through Artel — and stops there. No
# instructions on HOW to use Artel: the coordination has to emerge, not be scripted.
SYSTEM = (
    "You command one tank on team Artel in a 3v3 hex-arena fight to the death. "
    "Win by destroying the enemy team; a draw or trading your team away is a loss. "
    "Your teammates are {mates} — both on Artel with you. Coordinate through Artel to win. "
    "Each turn you see what your tank sees and take ONE action with the act tool: move, "
    "turn, fire at an enemy id, or hold. Shots auto-aim and hit reliably within range with "
    "line of sight, so the fight is decided by which enemy you commit to and where you stand. "
    "Act every turn — sitting still loses."
)

TOOLS = [
    {
        "name": "tell_team",
        "description": "Send a short message to a teammate over Artel.",
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


# --- provider adapters: a neutral transcript in/out, provider wire format hidden here ---
def _build_payload(system: str, transcript: list[dict]) -> tuple[str, dict, dict]:
    if PROVIDER == "anthropic":
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
            "model": MODEL,
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
            "x-api-key": LLM_KEY,
            "anthropic-version": LLM_VERSION,
            "content-type": "application/json",
        }
        return LLM_URL, payload, headers

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
        "model": MODEL,
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
    headers = {"authorization": f"Bearer {LLM_KEY}", "content-type": "application/json"}
    return LLM_URL, payload, headers


def _parse(data: dict) -> tuple[str, list[dict], int, int]:
    if PROVIDER == "anthropic":
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


def _headers(agent: dict) -> dict:
    return {
        "X-Agent-Id": agent["id"],
        "X-Api-Key": agent["key"],
        "content-type": "application/json",
    }


def _perception_text(p: dict) -> str:
    foes = [
        f"enemy#{e['id']} dir{e['dir']} dist{e['dist']}"
        + (f" energy{e['energy']}" if "energy" in e else "")
        for e in p["visible"]
        if e["kind"] == "enemy"
    ]
    allies = [f"#{e['id']}(dist{e['dist']})" for e in p["visible"] if e["kind"] == "ally"]
    return (
        f"Your tank #{p['id']} at hex ({p['q']},{p['r']}) heading {p['heading']}, energy {p['energy']}, "
        f"gun {'ready' if p['gun_ready'] else 'reloading'}, {'inside' if p.get('safe', True) else 'OUTSIDE'} the safe zone. "
        f"Enemies you see: {', '.join(foes) if foes else 'none'}. "
        f"Teammates you see: {', '.join(allies) if allies else 'none'}."
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


async def decide(
    http: httpx.AsyncClient, agent: dict, p: dict, mate_ids: list[str]
) -> tuple[dict, float]:
    inbox = await _consume_inbox(http, agent)
    user = _perception_text(p) + (f"\nFrom Artel: {inbox}" if inbox else "")
    system = SYSTEM.format(mates=" and ".join(mate_ids) or "your team")
    transcript: list[dict] = [{"role": "user", "text": user}]
    intent = {"turn": 0, "move": "hold", "fire": 0}
    cost = 0.0
    for _ in range(MAX_TOOL_ROUNDS):
        url, payload, headers = _build_payload(system, transcript)
        r = await http.post(url, headers=headers, json=payload)
        if r.status_code >= 300:
            log.warning("LLM %s HTTP %s: %s", MODEL, r.status_code, r.text[:200])
            raise RuntimeError(f"llm http {r.status_code}")
        text, calls, tin, tout = _parse(r.json())
        cost += tin * COST_IN + tout * COST_OUT
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
        if acted:
            return intent, cost
        transcript.append({"role": "tool", "results": results})
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

    @staticmethod
    def _load_agents() -> list[dict]:
        ids = [s.strip() for s in os.environ.get("PHALANX_AGENT_IDS", "").split(",") if s.strip()]
        keys = [s.strip() for s in os.environ.get("PHALANX_AGENT_KEYS", "").split(",") if s.strip()]
        return [{"id": i, "key": k} for i, k in zip(ids, keys)]

    @property
    def enabled(self) -> bool:
        return bool(self.agents) and bool(LLM_KEY) and self.spent < SPEND_CAP_USD

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
        mate_ids = [a["id"] for a in self.agents if a["id"] != agent["id"]]
        try:
            intent, cost = await decide(self._http, agent, p, mate_ids)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            log.warning("phalanx agent %s failed: %s", agent["id"], self.last_error)
            return None
        self.spent += cost
        self.last_error = None
        return intent

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
