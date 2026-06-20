from __future__ import annotations

from llmrouter import Request

from verglas.engine import INTEGRITY_FREE_DARK, Game
from verglas.meeting import COLD_PERSONA, persona

# The autonomous task phase: each free (or just-interrupted) agent is asked for ONE action, as a tool
# call. The tools are deliberately dead-simple — every target is an enum the model picks from a short
# given list (a room, a name), never a free-form id — so even the smaller free-tier models return a
# clean, valid call. Anything missing/malformed falls back to engine.default_action upstream.

SYS_CREW = (
    "You are {name}, overwintering at an isolated Antarctic station. One of the crew is secretly the Cold "
    "— it wears a friend's face, and it can ONLY kill in a room that has gone DARK. You all just have to "
    "SURVIVE THE STORM; it passes by dawn and you win. But the STATION ITSELF is dying wherever the lights "
    "are out — every dark room bleeds the outpost's integrity, and if it hits zero the whole station blacks "
    "out and you ALL lose, even untouched. So you cannot just huddle in one safe corner: the crew must "
    "SPREAD OUT and keep the WHOLE station lit. RELIGHT dark rooms — it denies the Cold a place to hunt AND "
    "keeps the station alive. Never end up alone in the dark — if you must cross a dark room, "
    "stick right beside another crewmate, because it can't strike with someone close enough to see. If "
    "someone whispers you to slip off into the dark together, that's exactly how it isolates you — weigh "
    "who's asking and whether the room is lit; you can refuse. Trust no claim you can't check. "
    "Pick exactly ONE action by calling a tool."
)
SYS_COLD = (
    "You are {name}. Secretly you ARE the Cold — something that came in from the storm, wearing a "
    "crewmate's face. The crew win if they SURVIVE THE STORM until dawn, so you must thin them before it "
    "passes. THE ONE HARD RULE: you can ONLY kill in a DARK room, right next to a crewmate, with no OTHER "
    "crewmate close enough to see it — in the dark, sight is short, so you don't need the room empty, just "
    "your victim isolated in a corner of the gloom. The dark is your weapon: SABOTAGE the lights (snuff "
    "your room or a neighbouring one) to MAKE a place to hunt, then WHISPER or FOLLOW a crewmate into it "
    "('help me in the Reactor', 'buddy up, just us two'). When you're beside one in the dark with no one "
    "else near, the eliminate option appears — TAKE it, then slip away (vent if you can). The rest of the "
    "time, blend in: move with the group, pretend to relight, look helpful and worried. Use WHISPERS to "
    "MANIPULATE — win a crewmate's trust to lure them off alone, sow distrust between two others, and set "
    "up the alibi you'll lean on later. Never loiter hoping for a chance — go MAKE the dark and draw "
    "someone into it. Pick exactly ONE action by calling a tool."
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
    dark_rooms = sorted(set(g.open_tasks))  # dark rooms waiting to be relit
    if dark_rooms:
        tools.append(
            _tool(
                "go_to_task",
                "Walk to a DARK room and restore its lights — denies the Cold a place to hunt.",
                {"room": _enum("which dark room to relight", dark_rooms)},
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
                    "Kill the crewmate you are alone with in the dark, right beside you.",
                    {"who": _enum("who to eliminate", kill_names)},
                    ["who"],
                )
            )
        if (
            g.sab_cd == 0
        ):  # snuff the lights to make a hunting ground (your room or a lit neighbour)
            lit = [
                r
                for r in dict.fromkeys([a.room, *sorted(g.adj.get(a.room, []))])
                if r not in g.dark
            ]
            if lit:
                tools.append(
                    _tool(
                        "darken",
                        "Kill the lights in your room or a neighbour — make a dark place to hunt.",
                        {"room": _enum("room to plunge into darkness", lit)},
                        ["room"],
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
    dark_s = ", ".join(sorted(g.dark)) or "none — the whole station is lit"
    alive, total = len(g.living()), len(g.agents)
    dead = total - alive
    here_dark = a.room in g.dark
    lines = [
        f"You are in the {a.room}, which is {'DARK' if here_dark else 'LIT'}. With you: {here_s}.",
        f"Dark rooms now (a kill can only happen in one of these, and each one bleeds the station): {dark_s}.",
        f"Crew alive: {alive} of {total}. Outlast the storm to dawn and the crew win.",
    ]
    if g.integrity_on:
        ndark = len(g.dark)
        state = (
            f"holding ({ndark} dark, within the {INTEGRITY_FREE_DARK}-room limit)"
            if ndark <= INTEGRITY_FREE_DARK
            else f"BLEEDING — {ndark} rooms dark, over the {INTEGRITY_FREE_DARK}-room limit"
        )
        lines.append(
            f"Station integrity: {int(g.integrity)}% and {state}. It drains once more than "
            f"{INTEGRITY_FREE_DARK} rooms are dark, and at 0 the whole crew LOSE. So don't huddle — "
            "spread out and relight to keep dark rooms under the limit."
        )
    # the Cold's situational read, straight off the engine's rule: dark + within reach + no crew watching.
    if a.impostor:
        kills = [g.by_id(i).name for i in g.legal_kills(a)]
        crew_here = [o.name for o in g._occ(a.room) if o.id != a.id and not o.impostor]
        if kills:
            lines.append(
                f"KILL WINDOW: {kills[0]} is right by you in the dark with no one close to see — strike NOW."
            )
        elif crew_here and not here_dark:
            lines.append(
                f"{crew_here[0]} is here but the room is LIT — DARKEN it (sabotage), then close in for the kill."
            )
        elif crew_here:
            lines.append(
                f"{crew_here[0]} is here in the dark but not isolated — get right beside them, away from the others."
            )
        else:
            lines.append(
                "No crewmate to hunt here. Find one drifting off, or darken a room and draw someone into it."
            )
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
    base = (SYS_COLD if a.impostor else SYS_CREW).format(name=a.name)
    # carry the agent's persona into the task phase too, so anything it whispers reads unmistakably as them
    voice = COLD_PERSONA if a.impostor else f"You are unmistakably {a.name}: {persona(a.name)}"
    sys = f"{base} {voice[0].upper() + voice[1:]}. Let that voice flavour anything you whisper."
    return Request(
        system=sys,
        user=_context(g, a, inbox or []),
        temperature=0.7,
        timeout=12.0,
        requires_tools=True,
        tools=build_tools(g, a),
    )
