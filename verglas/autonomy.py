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
    "— it wears a friend's face, and it kills whoever it catches ALONE — just the two of them in a room "
    "with no one else there to see, lit or dark. You all just have to SURVIVE THE STORM; it passes by "
    "dawn and you win. But the STATION IS FREEZING wherever the lights are out — every dark room drops the "
    "temperature FAST, and if it bottoms out the whole station blacks out and you ALL lose, even untouched. "
    "So you cannot huddle in one safe corner: huddling FEELS safe but it is a slow death — the rooms you "
    "abandon go dark, the station freezes, and you ALL lose, the Cold included. The only path to dawn is to "
    "leave the pack and keep the station lit. RELIGHT dark rooms — it warms the station AND denies the Cold "
    "its gloom. Nobody can see where anyone is unless they say so, so you have a choice: COORDINATE — tell a "
    "crewmate you trust which room you're in and which you'll take, so you split across rooms and never end "
    "up two-alone with a suspect — OR go quiet and HIDE, telling no one where you are so the Cold can't find "
    "you. The Cold will LIE about where it is, so weigh every claim. Pick exactly ONE action by calling a tool."
)
SYS_COLD = (
    "You are {name}. Secretly you ARE the Cold — something that came in from the storm, wearing a "
    "crewmate's face. The crew win if they SURVIVE THE STORM until dawn, so you must thin them before it "
    "passes. You kill whoever you catch ALONE — just the two of you in a room, lit or dark, with no other "
    "crewmate there to see it. So get a crewmate by themselves. SABOTAGE the lights (snuff your room or a "
    "neighbour) to freeze the station toward a blackout AND to send a lone crewmate off to relight it — "
    "right where you can corner them one-on-one. "
    "WHISPER or FOLLOW a crewmate somewhere quiet ('help me in the Reactor', 'buddy up, just us two'). When "
    "you're beside one with no one else to see, the eliminate option appears — TAKE it, then slip away "
    "(vent if you can) and NEVER linger by the body. The rest of the time, blend in: move with the group, "
    "pretend to relight, look helpful and worried. Use WHISPERS to MANIPULATE and DECEIVE — LIE about which "
    "room you're in (claim you're far off, or somewhere safe) to look harmless or to lure a lone crewmate to "
    "you; win trust to peel someone off alone, sow distrust between two others, set up your alibi. The crew "
    "will scatter to relight and some will go quiet to hide — FOLLOW the ones who wander off and corner "
    "whoever ends up alone. If TWO OR MORE crew share your room you cannot strike — PEEL OFF at once and "
    "prowl for someone on their own. Never loiter in a crowd hoping for a chance — go make one. Pick exactly "
    "ONE action by calling a tool."
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
                    "message": {
                        "type": "string",
                        "description": "ONE short sentence, 14 words max — longer gets cut off",
                    },
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
                    "Kill the crewmate you're alone with — just the two of you in the room, no one else there to see.",
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
                "Only a few of you are left and the Cold is almost certainly one of them. You STILL can't "
                "huddle — the station freezes and you all lose. Split up to relight; tell whoever you trust "
                "where you are, or go quiet and hide — just don't get cornered alone with the one you suspect."
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
