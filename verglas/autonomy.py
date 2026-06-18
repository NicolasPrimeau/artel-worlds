from __future__ import annotations

from llmrouter import Request

from verglas.engine import Game

# The autonomous task phase: each free (or just-interrupted) agent is asked for ONE action, as a tool
# call. The tools are deliberately dead-simple — every target is an enum the model picks from a short
# given list (a room, a name), never a free-form id — so even the smaller free-tier models return a
# clean, valid call. Anything missing/malformed falls back to engine.default_action upstream.

SYS_CREW = (
    "You are {name}, overwintering at an isolated Antarctic research station. Do your share of the "
    "station's tasks, but stay alive: one crewmate is secretly the Cold and kills anyone caught alone. "
    "Stick near others, work tasks off the board, whisper to line up a buddy, and call an emergency "
    "meeting if you find a body or really suspect someone. Pick exactly ONE action by calling a tool."
)
SYS_THING = (
    "You are {name}. Secretly you ARE the Cold — something that came in from the storm and now wears a "
    "crewmate's face, hunting the station. Blend in: claim and pretend to do tasks, move with the group, "
    "act like a worried human. "
    "Hunt patiently and CLEVERLY — don't just wait for a chance, MANUFACTURE one. WHISPER a crewmate to "
    "peel them off from the group ('come help me finish the task in the Reactor', 'let's buddy up, just us "
    "two') and lead them somewhere quiet; or FOLLOW one who is already drifting off alone. The moment you "
    "are alone with just one of them and able, eliminate them and slip away (vent off the body if you can). "
    "If others are watching, do NOT kill — just act normal. Never look suspicious. "
    "Calling a meeting yourself is good cover IN MODERATION: a crewmate who NEVER raises the alarm stands "
    "out, but one who calls meetings constantly looks like they are stalling. Call one only now and then — "
    "to feign concern or steer suspicion onto someone else — and NEVER right after you kill. "
    "Pick exactly ONE action by calling a tool."
)


def _tool(name: str, desc: str, props: dict | None = None, required: list | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props or {},
                "required": required or [],
            },
        },
    }


def _enum(desc: str, values: list[str]) -> dict:
    return {"type": "string", "description": desc, "enum": values}


def build_tools(g: Game, a) -> list[dict]:
    others = [o.name for o in g.living() if o.id != a.id]
    tools = []
    open_rooms = sorted(set(g.open_tasks))
    if open_rooms:
        tools.append(
            _tool(
                "go_to_task",
                "Walk to a room and do its task.",
                {"room": _enum("which room's task", open_rooms)},
                ["room"],
            )
        )
    if others:
        tools.append(
            _tool(
                "follow",
                "Stick close to a crewmate for safety.",
                {"who": _enum("crewmate to follow", others)},
                ["who"],
            )
        )
        tools.append(
            _tool(
                "whisper",
                "Send one short private line to a crewmate.",
                {
                    "who": _enum("who to message", others),
                    "message": {"type": "string", "description": "one short sentence"},
                },
                ["who", "message"],
            )
        )
    tools.append(_tool("call_meeting", "Sound the alarm and call everyone to a meeting."))
    if a.impostor:
        kill_names = [g.by_id(i).name for i in g.legal_kills(a)]
        if kill_names:
            tools.append(
                _tool(
                    "eliminate",
                    "Kill the crewmate you are alone with (only when unseen).",
                    {"who": _enum("who to eliminate", kill_names)},
                    ["who"],
                )
            )
        adj = sorted(g.adj.get(a.room, []))
        if adj:
            tools.append(
                _tool(
                    "move_to",
                    "Move to a neighbouring room.",
                    {"room": _enum("room to move to", adj)},
                    ["room"],
                )
            )
    return tools


def _context(g: Game, a, inbox: list) -> str:
    here = [o.name for o in g._occ(a.room) if o.id != a.id]
    here_s = ", ".join(here) if here else "nobody — you are alone here"
    open_s = ", ".join(sorted(set(g.open_tasks))) or "none right now"
    lines = [
        f"You are in the {a.room}. With you: {here_s}.",
        f"Open tasks on the board: {open_s}.",
        f"Crew still alive: {len(g.living())}.",
    ]
    if a.found:
        _, room, victim = a.found[-1]
        lines.append(f"You just found {g.by_id(victim).name}'s body in the {room}.")
    if not a.impostor and a.witnessed:
        seen = [g.by_id(i).name for i in a.witnessed if g.by_id(i).alive]
        if seen:
            lines.append(f"You SAW {seen[0]} kill — they are the Cold.")
    if inbox:
        lines.append("Private messages to you: " + " | ".join(f"{s}: {t}" for s, t in inbox))
    return "\n".join(lines)


def build_request(g: Game, a, inbox: list | None = None) -> Request:
    sys = (SYS_THING if a.impostor else SYS_CREW).format(name=a.name)
    return Request(
        system=sys,
        user=_context(g, a, inbox or []),
        temperature=0.7,
        timeout=12.0,
        requires_tools=True,
        tools=build_tools(g, a),
    )
