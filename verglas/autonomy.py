from __future__ import annotations

from llmrouter import Request

from verglas.engine import Game

# The autonomous task phase: each free (or just-interrupted) agent is asked for ONE action, as a tool
# call. The tools are deliberately dead-simple — every target is an enum the model picks from a short
# given list (a room, a name), never a free-form id — so even the smaller free-tier models return a
# clean, valid call. Anything missing/malformed falls back to engine.default_action upstream.

SYS_CREW = (
    "You are {name}, overwintering at an isolated Antarctic research station. One of the crew is secretly "
    "the Cold — it wears a friend's face and kills anyone it gets alone. You win by finishing every task "
    "on the board before it picks you off, or by voting it out once a body is found. Work the board and "
    "keep where others can see you. When someone asks you to slip off together 'to buddy up', that is "
    "exactly how the Cold gets a victim one-on-one — weigh WHO is asking and whether others are near; you "
    "can refuse and stay with the group. Trust no claim you can't check. "
    "Pick exactly ONE action by calling a tool."
)
SYS_THING = (
    "You are {name}. Secretly you ARE the Cold — something that came in from the storm and now wears a "
    "crewmate's face, hunting the station. Blend in: claim and pretend to do tasks, move with the group, "
    "act like a worried human; never look suspicious. "
    "You are on BORROWED TIME: the crew win the instant they finish every task on the board (a meter you "
    "cannot touch directly), and they will vote you out if they pin you. So hunt, and hunt soon — thin "
    "them faster than the tasks get done, before you're cornered. Hunt CLEVERLY: MANUFACTURE the chance. "
    "WHISPER a crewmate to peel them off ('come help me finish the task in the Reactor', 'let's buddy up, "
    "just us two') and lead them somewhere quiet, or FOLLOW one already drifting off alone. The instant "
    "you are alone with just one of them and able, eliminate them and slip away (vent off the body if you "
    "can). If anyone else is watching, do NOT kill — just act normal. "
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
    alive, total = len(g.living()), len(g.agents)
    dead = total - alive
    lines = [
        f"You are in the {a.room}. With you: {here_s}.",
        f"Open tasks on the board: {open_s}.",
        f"Crew still alive: {alive} of {total}.",
    ]
    # the dread grows with the body count — crew get more unsettled and trust the room less
    if not a.impostor and dead:
        if dead >= 4:
            lines.append(
                "Most of the crew are gone. The Cold is almost certainly someone still standing here — "
                "take nothing on faith and don't let anyone walk you off alone."
            )
        elif dead >= 2:
            lines.append(
                f"{dead} are dead now. Whoever's left could be it — keep witnesses close and be wary of "
                "anyone steering you somewhere quiet."
            )
        else:
            lines.append(
                "Someone is dead. Stay where others can see you and watch who you're alone with."
            )
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
