from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

# Verglas — a faithful Among Us shape. N agents walk a ship of rooms. Crew work a shared task board; the
# IMPOSTOR kills on a cooldown and can vent to slip away from a body. Vision is LOCAL: you only see who
# shares your room, so every agent leaves the task phase with a DIFFERENT, partial log of who-was-where.
# A found body (or the emergency button) calls a meeting: everyone reconvenes, pools what they saw, and
# votes someone out. Crew win by clearing every task OR ejecting every impostor; the Cold wins only by
# taking the LAST crewmate — no parity shortcut, so the endgame plays out as a final hunt.
#
# Two phases, two Artel primitives. The task board is Artel TASKS (claim/complete shared work); the
# meeting runs on Artel EVENTS/MESSAGES (pool testimony). The engine here is offline-pure — per-agent
# counters and a `decide_votes` callback — so the same board drives the deterministic A/B baseline or
# LLM agents. The live runner mirrors task completion and testimony onto a real Artel server.

# An Antarctic research station, blizzard outside, nobody leaving. One of the team isn't human anymore
# — the Cold wears a friend's face. The Mess Hall is the hub where everyone reconvenes for a meeting.
HUB = "Mess Hall"
# The 12 rooms of the outpost. Their floorplan and connections are generated fresh every game
# (_generate_station) — a rectangle recursively sliced into abutting rooms, never the same twice — so
# no two outposts look or play alike. 12 rooms is enough that the Cold can get a victim ALONE, so most
# kills have no eyewitness and the meeting turns on deduction (alibis, who-was-where), not "I saw it".
ROOM_NAMES = [
    "Bunks",
    "Infirmary",
    "Lab",
    "Greenhouse",
    "Comms",
    HUB,
    "Storage",
    "Reactor",
    "Garage",
    "Drill Site",
    "Generator",
    "Freezer",
]
# rooms get their names by size: the biggest slice is the dining hall, the smallest a cold-locker.
SIZE_ORDER = [
    HUB,
    "Garage",
    "Drill Site",
    "Greenhouse",
    "Bunks",
    "Lab",
    "Reactor",
    "Storage",
    "Generator",
    "Infirmary",
    "Comms",
    "Freezer",
]
GW, GH = 48, 34  # base floorplan grid for a full 12-room station; scaled down for smaller crews
_ROOM_MIN = 7  # min room dimension in tiles (keeps rooms reasonable on the fine grid)
_MIN_DOOR = 6  # min shared-wall length (tiles) for an EXTRA (loop) doorway
ROOMS_PER_AGENT = 1.3  # the outpost scales with the crew; ~9 rooms for the default 7 — sparse enough for the Cold to hunt
MIN_ROOMS = 6  # never fewer than this, or there's nowhere to isolate a kill or hide


def _room_count(n_agents: int) -> int:
    return max(MIN_ROOMS, min(len(ROOM_NAMES), round(n_agents * ROOMS_PER_AGENT)))


def _generate_station(rng: random.Random, n_rooms: int = 12):
    # BSP dungeon layout (how games fake real building plans): recursively partition the grid, place a
    # room INSET inside each leaf cell (varied size + position, with gaps between rooms), then connect
    # the rooms with L-shaped corridors. Result reads as architecture — clean rectangular rooms scattered
    # in a rectilinear footprint, linked by corridors — not a grid, not blobs, not a comb.
    # the grid scales with the target room count (density held ~constant), so a smaller crew gets a
    # smaller, tighter outpost — same room geometry, just fewer of them.
    scale = (n_rooms / 12) ** 0.5
    gw = max(2 * _ROOM_MIN + 6, round(GW * scale))
    gh = max(2 * _ROOM_MIN + 6, round(GH * scale))
    s = 2 * _ROOM_MIN  # a cell must be this big in a dim to split (leaving two >= _ROOM_MIN)
    cells = [(1, 1, gw - 2, gh - 2)]
    while len(cells) < n_rooms:
        cands = sorted(
            (i for i in range(len(cells)) if cells[i][2] >= s or cells[i][3] >= s),
            key=lambda i: -cells[i][2] * cells[i][3],
        )
        if not cands:
            break
        x, y, w, h = cells.pop(cands[rng.randrange(min(3, len(cands)))])
        if w >= s and (h < s or w >= h or rng.random() < 0.5):
            c = rng.randint(_ROOM_MIN, w - _ROOM_MIN)
            cells += [(x, y, c, h), (x + c, y, w - c, h)]
        else:
            c = rng.randint(_ROOM_MIN, h - _ROOM_MIN)
            cells += [(x, y, w, c), (x, y + c, w, h - c)]
    rooms = []
    for x, y, w, h in cells:
        # inset the room with a GUARANTEED >=1-tile margin on every side, so rooms in adjacent cells can
        # never touch. Abutting rooms would share a walkable border with no drawn door — reading as a
        # sealed cluster and letting agents cross the wall. A gap forces every link to be a real corridor.
        rw = min(w - 2, max(_ROOM_MIN - 2, w - rng.randint(1, max(1, w // 3))))
        rh = min(h - 2, max(_ROOM_MIN - 2, h - rng.randint(1, max(1, h // 3))))
        ox = rng.randint(1, w - rw - 1) if w - rw - 1 >= 1 else 1
        oy = rng.randint(1, h - rh - 1) if h - rh - 1 >= 1 else 1
        rooms.append((x + ox, y + oy, rw, rh))
    order = sorted(
        range(len(rooms)), key=lambda i: -rooms[i][2] * rooms[i][3]
    )  # biggest -> Mess Hall
    names = list(SIZE_ORDER[: len(rooms)])
    rects = {SIZE_ORDER[k]: rooms[order[k]] for k in range(len(rooms))}
    cen = {n: (rects[n][0] + rects[n][2] / 2, rects[n][1] + rects[n][3] / 2) for n in names}
    roomtiles = {
        (tx, ty)
        for (x, y, w, h) in rects.values()
        for ty in range(y, y + h)
        for tx in range(x, x + w)
    }
    adj: dict[str, set] = {n: set() for n in names}
    doors: dict = {}
    corr: set = set()

    def carve(a, b):
        ax, ay, bx, by = int(cen[a][0]), int(cen[a][1]), int(cen[b][0]), int(cen[b][1])
        bend = (bx, ay) if rng.random() < 0.5 else (ax, by)
        leg1 = (
            [(t, ay) for t in range(min(ax, bend[0]), max(ax, bend[0]) + 1)]
            if bend[1] == ay
            else [(ax, t) for t in range(min(ay, bend[1]), max(ay, bend[1]) + 1)]
        )
        leg2 = (
            [(t, by) for t in range(min(bx, bend[0]), max(bx, bend[0]) + 1)]
            if bend[1] == by
            else [(bx, t) for t in range(min(by, bend[1]), max(by, bend[1]) + 1)]
        )
        for x, y in leg1 + leg2:
            for dx in (0, 1):
                if 0 <= x + dx < gw and 0 <= y < gh and (x + dx, y) not in roomtiles:
                    corr.add((x + dx, y))
        adj[a].add(b)
        adj[b].add(a)
        doors[(a, b)] = (bend[0] + 0.5, bend[1] + 0.5)

    # MST over room centres → connected corridor tree; then a few extra links for loops
    intree = {names[0]}
    while len(intree) < len(names):
        best = min(
            ((cen[a][0] - cen[b][0]) ** 2 + (cen[a][1] - cen[b][1]) ** 2, a, b)
            for a in intree
            for b in names
            if b not in intree
        )
        carve(best[1], best[2])
        intree.add(best[2])
    extra = sorted(
        ((cen[a][0] - cen[b][0]) ** 2 + (cen[a][1] - cen[b][1]) ** 2, a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if b not in adj[a]
    )
    for _, a, b in extra[:6]:
        if rng.random() < 0.4:
            carve(a, b)
    vents: dict[str, list] = {}
    nonadj = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if b not in adj[a]]
    rng.shuffle(nonadj)
    for a, b in nonadj[: rng.randint(2, 3)]:
        vents.setdefault(a, []).append(b)
        vents.setdefault(b, []).append(a)

    # GUARANTEE one connected walkable network. The MST connects room ADJACENCY, but the geometric
    # corridor can leave the floorplan split into clusters with no walkable link — the renderer's path
    # router then can't route between them and an agent straight-lines through a wall. Repeatedly find a
    # disconnected component and carve an open-space path from it to the rest (BFS through empty tiles
    # only, never tunnelling through another room), until every room+corridor tile is one component.
    roomtile_set = set(roomtiles)

    def _components():
        walk = corr | roomtile_set
        comp, cid = {}, 0
        for t in walk:
            if t in comp:
                continue
            comp[t] = cid
            stack = [t]
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (x + dx, y + dy)
                    if nb in walk and nb not in comp:
                        comp[nb] = cid
                        stack.append(nb)
            cid += 1
        return comp, cid

    for _ in range(24):
        comp, ncomp = _components()
        if ncomp <= 1:
            break
        walk = corr | roomtile_set
        prev = {t: None for t in walk if comp[t] == 0}  # BFS frontier = all of component 0
        dq = deque(prev)
        target = None
        while dq:
            x, y = dq.popleft()
            if (x, y) in walk and comp.get((x, y), 0) != 0:
                target = (x, y)
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if not (0 <= nb[0] < gw and 0 <= nb[1] < gh) or nb in prev:
                    continue
                if nb in walk or nb not in roomtile_set:  # walkable, or empty space we may carve
                    prev[nb] = (x, y)
                    dq.append(nb)
        if target is None:
            break
        node = target
        while node is not None:
            if node not in roomtile_set:  # carve only the open-space tiles into corridor
                corr.add(node)
            node = prev[node]

    centers = {n: (round(cen[n][0], 2), round(cen[n][1], 2)) for n in names}
    return names, {k: sorted(v) for k, v in adj.items()}, vents, rects, doors, centers, sorted(corr)


TASKS_EACH = 5
TASK_P = (
    0.3  # per-tick chance an idle crew CLAIMS an open task off the board (then works WORK_TICKS)
)
TASK_SPAWN_P = (
    0.5  # per-tick chance a fresh task appears on the shared board (capped at one per crew)
)
WORK_TICKS = 3  # a task occupies its crew for several ticks — they linger at the console
TASK_SLACK = 4  # keep this many fewer tasks-in-play than living players, so a couple are always free to buddy
KILL_CD = 3  # ~15s between kills (each task tick runs ~5s, so 3 ticks ≈ 15s)
KILL_REACH = 3.0  # the Cold must be within this many cells of the victim — no killing across a room
WITNESS_RANGE = 5.0  # in the dark, a kill is only seen by crew within this many cells of the victim
# --- the storm & the dark ---------------------------------------------------------------------------
# Light is safety: the Cold can ONLY kill in a DARK room. The storm is the clock — survive it and the
# crew win. It darkens rooms over time (more, faster, late); crew relight them (the task board); the Cold
# can darken rooms itself (sabotage) to manufacture a kill spot. These are all tunable.
STORM_TICKS = 70  # game length (~5 min at ~4.2s/tick) — survive to here and the crew win
STORM_EVERY = (
    3  # the storm darkens a lit room about this often (ticks); it speeds up late (see _storm).
    # kept brisk so the relight board stays stocked — crew keep moving to tasks instead of bunching idle
)
DARK_CAP = (
    8  # the storm won't push past this many dark rooms (the Cold's sabotage can still add more)
)
SABOTAGE_CD = (
    4  # ticks between the Cold's light-sabotages (~16s) — its main tool for making kill spots
)
START_DARK = (
    6  # rooms already dark when the game opens, so the outpost reads as embattled from minute one
)
GX_STEP = (
    1.6  # cells/tick an agent drifts toward its in-room spot (the Cold stalks into range gradually)
)
# spread spots around a room centre (unit offsets) so co-located agents stand apart, not stacked at centre
_SPREAD = [
    (0, 0),
    (-1, -1),
    (1, -1),
    (1, 1),
    (-1, 1),
    (0, -1.4),
    (1.4, 0),
    (0, 1.4),
    (-1.4, 0),
    (0.7, -0.7),
]
START_GRACE = KILL_CD  # the same ~15s gates the FIRST kill too — no long opening grace
FOLLOW_TICKS = 6  # how long an autonomous agent tails a buddy before it stops to decide again
# --- station integrity ------------------------------------------------------------------------------
# the outpost is dying where the lights are out: every DARK room bleeds integrity each tick. Let it hit
# zero and the station blacks out — the crew lose even if alive. This FORCES them to spread out and keep
# the whole station lit instead of huddling in a safe corner. Relighting (clearing the board) is the cure.
INTEGRITY_MAX = 100.0
INTEGRITY_FREE_DARK = 4  # the station tolerates up to this many dark rooms; only the EXCESS bleeds it (scaled per game)
DARK_DRAIN = 1.2  # temperature lost per tick per dark room BEYOND the allowance — menacing: dark rooms bite fast
INTEGRITY_RECOVER = 1.2  # regained per tick while at/under the allowance — slow enough that slacking genuinely costs you
# crew survive to dawn ~97%, a huddling crew blacks out ~97% — relight enough and you live, slack and you don't
# meetings happen ONLY on a body report — there is no emergency button (no calling a meeting with no body)
MAX_TICKS = 600

# the winter-over crew — every AI that ever got a NAME: real assistants and fictional machine minds. The
# joke tells itself — a fleet of named AIs iced in at the outpost, and one of them is quietly the Cold:
# the exact "the station's own computer is not on our side" trope these names come from (HAL, Ash, GERTY,
# Mother). Each game samples a distinct subset from the seed, so names overlap between games but never
# repeat within one (pitch-style). ~60 names, 10 seats a game → you rarely see more than a face or two
# carry over from the last round.
NAMES = [
    # real assistants, chatbots & named systems
    "Alexa",  # Amazon
    "Siri",  # Apple
    "Cortana",  # Microsoft (also Halo)
    "Bixby",  # Samsung
    "Claude",  # Anthropic
    "Watson",  # IBM
    "Copilot",  # GitHub
    "Bard",  # Google
    "Gemini",  # Google
    "Ernie",  # Baidu
    "Clippy",  # Microsoft Office
    "Eliza",  # Weizenbaum — the first chatbot
    "Tay",  # Microsoft's ill-fated bot
    "Sydney",  # Bing's alter ego
    "LaMDA",  # Google
    "Meena",  # Google
    "Sparrow",  # DeepMind
    "Replika",  # Luka
    "Cleverbot",  # Carpenter
    "Mitsuku",  # Kuki / Pandorabots
    "Sophia",  # Hanson Robotics
    "Megatron",  # NVIDIA (and the Transformer)
    "Galactica",  # Meta
    "Mycroft",  # open-source assistant (and Heinlein's "Mike")
    "Whisper",  # OpenAI
    "Deep Blue",  # IBM — beat Kasparov
    # fictional machine minds
    "HAL",  # 2001: A Space Odyssey
    "GLaDOS",  # Portal
    "Skynet",  # Terminator
    "GERTY",  # Moon
    "Mother",  # Alien (MU-TH-UR)
    "Ash",  # Alien
    "Bishop",  # Aliens
    "Samantha",  # Her
    "TARS",  # Interstellar
    "CASE",  # Interstellar
    "Jarvis",  # Iron Man
    "FRIDAY",  # Iron Man
    "Ultron",  # Avengers
    "Vision",  # Avengers
    "KITT",  # Knight Rider
    "Marvin",  # The Hitchhiker's Guide to the Galaxy
    "Data",  # Star Trek
    "Ava",  # Ex Machina
    "Dolores",  # Westworld
    "Wintermute",  # Neuromancer
    "SHODAN",  # System Shock
    "Multivac",  # Asimov
    "Holly",  # Red Dwarf
    "EDI",  # Mass Effect
    "Legion",  # Mass Effect
    "WALL-E",  # WALL-E
    "EVE",  # WALL-E
    "VIKI",  # I, Robot
    "Sonny",  # I, Robot
    "Optimus",  # Transformers
    "Samaritan",  # Person of Interest
    "Joshua",  # WarGames
    "Colossus",  # Colossus: The Forbin Project
    "Bender",  # Futurama
    "Rosie",  # The Jetsons
    "K9",  # Doctor Who
    "Ziggy",  # Quantum Leap
    "Cylon",  # Battlestar Galactica
]


@dataclass
class Sighting:
    tick: int
    room: str
    present: tuple  # other agent ids co-located this tick


@dataclass
class Agent:
    id: int
    name: str
    impostor: bool
    room: str
    alive: bool = True
    tasks: int = TASKS_EACH  # this agent's slice of the shared Artel task board
    model: str = ""  # which LLM drives this agent in the meeting (set by the meeting layer)
    tasking: bool = False  # working a task this tick (transient, for the renderer)
    work: int = 0  # ticks left on the current task — while >0 the crew stays put at the console
    dest: str | None = None  # the room of a claimed task it's walking to (None = nothing to do)
    goto: str | None = None  # a plain move-to-room intent (no task) — used by autonomous agents
    follow: int | None = None  # an agent id this one is tailing (buddy up) — autonomous intent
    follow_ticks: int = 0  # ticks left tailing before the agent re-decides (follow isn't forever)
    gx: float = (
        -1.0
    )  # cell position WITHIN the room (engine-authoritative; the renderer draws from it)
    gy: float = -1.0  # so a kill can require the Cold to actually be close, not across the room
    # what this agent privately observed — the raw material for its testimony in a meeting
    trail: list = field(default_factory=list)  # (tick, room) — its own movements this round
    seen: list = field(default_factory=list)  # list[Sighting]
    witnessed: set = field(default_factory=set)  # impostor ids it directly saw kill
    found: list = field(default_factory=list)  # (tick, room, victim_id) bodies it discovered


@dataclass
class Meeting:
    tick: int
    reporter: int  # who called it (-1 = emergency button)
    room: str  # where the body was (or the hub for an emergency alarm)
    victim: int | None  # who died, if a body report
    ejected: int | None = None  # who got voted out
    votes: dict = field(default_factory=dict)  # voter id -> target id (or -1 for skip)
    transcript: list = field(default_factory=list)  # [(speaker_id, text)] — the meeting chat


@dataclass
class Game:
    rng: random.Random
    agents: list
    rooms: list = field(default_factory=list)  # this game's room names
    adj: dict = field(default_factory=dict)  # generated connections (doored)
    vents: dict = field(default_factory=dict)  # generated crawlspaces
    rects: dict = field(default_factory=dict)  # name -> (x,y,w,h) module rectangle (tiles)
    doors: dict = field(default_factory=dict)  # (a,b) -> (x,y) corridor connection point
    centers: dict = field(default_factory=dict)  # name -> (cx,cy) module centre in tiles
    corridor: list = field(default_factory=list)  # corridor tiles [(x,y), ...] linking the rooms
    outpost: int = 31  # the station's number (randomized per game)
    bodies: dict = field(default_factory=dict)  # room -> victim id, undiscovered
    tick: int = 0
    cd: int = 0  # shared impostor kill cooldown
    sab_cd: int = 0  # the Cold's light-sabotage cooldown
    dark: set = field(
        default_factory=set
    )  # rooms currently unlit — the only places a kill can happen
    winner: str | None = None  # "crew" | "impostor"
    win_by: str | None = None  # "storm" | "ejection" | "extinction" | "blackout"
    storm_by_ticks: bool = (
        True  # offline/tests end the night on STORM_TICKS; the live server runs a
    )
    # real-seconds dawn clock (the HUD "To Dawn") and disables this so the two never disagree
    integrity: float = (
        INTEGRITY_MAX  # the station's health — drains while rooms are dark (live only)
    )
    integrity_on: bool = (
        False  # live turns this on; off for offline/tests so their outcomes are unchanged
    )
    dark_cap: int = DARK_CAP  # storm's max dark rooms — scaled to the station size in new_game
    free_dark: int = (
        INTEGRITY_FREE_DARK  # dark rooms tolerated before integrity bleeds — scaled too
    )
    hunting: bool = False  # the final hunt is on: one crew left, the Cold has dropped the mask
    hunt_ticks: int = 0  # how long the final hunt has run (the flee can't last forever)
    meetings: list = field(default_factory=list)
    ejected_impostors: int = 0
    wrong_ejections: int = 0
    last_kill: dict | None = None  # {tick, victim, room} — transient signal for the kill animation
    open_tasks: list = field(
        default_factory=list
    )  # rooms with an unclaimed task console (the board)
    tasks_done: int = 0  # tasks completed this game (cumulative)
    tasks_goal: int = 0  # completions the crew need to win by tasks
    events: list = field(
        default_factory=list
    )  # task lifecycle this step → mirrored onto Artel tasks

    def living(self, impostor=None):
        return [a for a in self.agents if a.alive and (impostor is None or a.impostor == impostor)]

    def by_id(self, i):
        return next(a for a in self.agents if a.id == i)

    def _occ(self, room):
        return [a for a in self.living() if a.room == room]

    def _near(self, a, b) -> bool:
        return abs(a.gx - b.gx) + abs(a.gy - b.gy) <= KILL_REACH

    def _unwitnessed(self, m, victim) -> bool:
        # the Cold needs the victim with no CREW watching. In a LIT room anyone else present sees it, so the
        # victim must be alone; in the DARK only crew within sight of the victim witness, so it can strike in
        # a corner away from them. (An impostor ally never counts as a witness.)
        others = [c for c in self._occ(victim.room) if c.id != victim.id and not c.impostor]
        if victim.room in self.dark:
            return not any(
                abs(c.gx - victim.gx) + abs(c.gy - victim.gy) <= WITNESS_RANGE for c in others
            )
        return not others

    def _place(self) -> None:
        # update each living agent's cell within its room: a follower closes on whoever it's tailing (the
        # Cold stalks a victim this way), everyone else drifts to a spread spot around the room centre.
        # Movement is gradual (GX_STEP/tick), so closing the gap takes ticks and reads on screen.
        byroom: dict = {}
        for a in self.living():
            byroom.setdefault(a.room, []).append(a)
        for room, occ in byroom.items():
            x, y, w, h = self.rects[room]
            cx, cy = x + (w - 1) / 2, y + (h - 1) / 2
            rad = max(1.0, min(w, h) * 0.26)
            occ.sort(key=lambda a: a.id)
            for i, a in enumerate(occ):
                if not (x < a.gx < x + w - 1 and y < a.gy < y + h - 1):  # just arrived / unplaced
                    a.gx, a.gy = cx, cy
                tgt = self.by_id(a.follow) if a.follow is not None else None
                if tgt is not None and tgt.alive and tgt.room == room:
                    gxg, gyg = tgt.gx, tgt.gy  # close on the one you're tailing
                else:
                    ox, oy = _SPREAD[i % len(_SPREAD)]
                    gxg, gyg = cx + ox * rad, cy + oy * rad
                a.gx += max(-GX_STEP, min(GX_STEP, gxg - a.gx))
                a.gy += max(-GX_STEP, min(GX_STEP, gyg - a.gy))
                a.gx = min(max(a.gx, x + 0.6), x + w - 1.6)
                a.gy = min(max(a.gy, y + 0.6), y + h - 1.6)

    def tasks_left(self):
        return max(0, self.tasks_goal - self.tasks_done)

    def _room_dist(self, ra, rb):
        ca, cb = self.centers[ra], self.centers[rb]
        return (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2

    def _toward_room(self, a, room):
        # step one room closer to `room` along the corridors (greedy on centre distance)
        if a.room == room:
            return
        a.room = min(self.adj[a.room], key=lambda r: self._room_dist(r, room))

    def _area(self, room):
        x, y, w, h = self.rects[room]
        return w * h

    def _task_cap(self, room):
        # bigger rooms hold more consoles, but never more than 2 active in any one room at a time
        return min(2, max(1, round(self._area(room) / 70)))

    def _darken(self, room) -> None:
        # snuff a room's lights → it becomes a kill spot AND a relight job on the board (source anonymous)
        if room not in self.dark:
            self.dark.add(room)
            if room not in self.open_tasks:
                self.open_tasks.append(room)
            self.events.append(("dark", room))

    def _storm(self) -> None:
        # the storm snuffs a random lit room; capped so it never blacks the whole station out at once
        if len(self.dark) >= self.dark_cap:
            return
        lit = [r for r in self.rooms if r not in self.dark]
        if lit:
            self._darken(self.rng.choice(lit))

    def _storm_due(self) -> bool:
        # the storm bites more often as the long night wears on (every ~5 ticks → every ~2 near the end)
        every = max(2, STORM_EVERY - self.tick // 24)
        return self.tick % every == 0

    def _tick_integrity(self) -> None:
        # the station bleeds out where it's dark; a fully lit station slowly recovers. Forces the crew to
        # spread out and keep the WHOLE outpost lit, not huddle in a safe corner while the rest goes dark.
        if not self.integrity_on:
            return
        excess = len(self.dark) - self.free_dark
        if excess > 0:
            self.integrity = max(0.0, self.integrity - DARK_DRAIN * excess)
        else:
            self.integrity = min(INTEGRITY_MAX, self.integrity + INTEGRITY_RECOVER)

    def sabotage(self, a, room) -> bool:
        # the Cold kills the lights in its own room or a neighbouring one (to make a kill spot). Anonymous.
        if not (a.impostor and self.sab_cd == 0):
            return False
        if (room != a.room and room not in self.adj.get(a.room, ())) or room in self.dark:
            return False
        self._darken(room)
        self.sab_cd = SABOTAGE_CD
        return True

    def _claim_task(self, a):
        # take the nearest DARK room off the board and head over to relight it
        room = min(self.open_tasks, key=lambda r: self._room_dist(a.room, r))
        self.open_tasks.remove(room)
        a.dest = room
        self.events.append(("claim", a.id, room))

    def _approach_buddy(self, a):
        # nothing to do → close on the nearest other survivor (safety in numbers); stay if already together
        others = [o for o in self.living() if o.id != a.id]
        if others:
            self._toward_room(a, min(others, key=lambda o: self._room_dist(a.room, o.room)).room)

    # a meeting happens ONLY when a CREW member finds a body — never the Cold standing over its own kill
    # (it's alone with the victim by the time it strikes, so it would otherwise always report). the corpse
    # waits in the dark until a crewmate walks in: the intended discovery beat. shared by step()/execute().
    def _report_body(self) -> Meeting | None:
        for room, victim in list(self.bodies.items()):
            finders = [a for a in self.living(impostor=False) if a.room == room]
            if finders:
                del self.bodies[room]
                for f in finders:
                    f.found.append((self.tick, room, victim))
                return Meeting(self.tick, finders[0].id, room, victim)
        return None

    # --- task phase: one tick of move / task / kill. Returns a Meeting trigger or None. ---
    def step(self) -> Meeting | None:
        self.tick += 1
        self.events = []  # fresh per tick; the live server drains these onto Artel after each step
        if self.cd > 0:
            self.cd -= 1
        if self.living(impostor=True) and len(self.living(impostor=False)) <= 1:
            self._final_hunt()  # one crew left → the scripted final hunt takes over the tick
            self._check_win()
            return None

        # the storm snuffs lit rooms over the night (faster late); the Cold's sabotage cooldown ticks
        if self.sab_cd > 0:
            self.sab_cd -= 1
        if self._storm_due():
            self._storm()
        self._tick_integrity()

        for a in self.living():
            a.tasking = False
            if a.impostor:
                # offline heuristic Cold (the live one is the LLM): if alone with one crewmate, darken the
                # room to make the kill (the kill block below then takes them); otherwise stalk the nearest
                # crewmate to engineer that one-on-one.
                occ = self._occ(a.room)
                crew_here = [c for c in occ if not c.impostor]
                if len(occ) == 2 and crew_here:
                    if a.room not in self.dark and self.sab_cd == 0:
                        self.sabotage(a, a.room)  # darken it — the kill block takes them this tick
                else:
                    # stalk a crewmate who is ALONE in their room (the only crew there), preferring the dark
                    lone = [
                        c
                        for c in self.living(impostor=False)
                        if sum(1 for o in self._occ(c.room) if not o.impostor) == 1
                    ]
                    pool = lone or self.living(impostor=False)
                    if pool:
                        tgt = min(
                            pool,
                            key=lambda c: (
                                c.room not in self.dark,
                                self._room_dist(a.room, c.room),
                            ),
                        )
                        self._toward_room(a, tgt.room)
            elif a.work > 0:  # mid-task: stay at the console until it's done
                a.work -= 1
                a.tasking = True
                if a.work == 0:
                    self.tasks_done += 1  # COMPLETE — one off the board
                    self.dark.discard(a.room)  # relit — the room is safe again
                    a.dest = None
                    self.events.append(("complete", a.id))
            elif a.dest is not None:  # walking to a claimed task
                if a.room == a.dest:
                    a.work = WORK_TICKS - 1  # arrived — start working the console
                    a.tasking = True
                    if a.work == 0:
                        self.tasks_done += 1
                        self.dark.discard(a.room)  # relit — the room is safe again
                        a.dest = None
                        self.events.append(("complete", a.id))
                else:
                    self._toward_room(a, a.dest)
            elif self.open_tasks:
                self._claim_task(a)  # something to do → claim the nearest task and set off for it
                if a.room == a.dest:
                    a.work = WORK_TICKS - 1
                    a.tasking = True
                    if a.work == 0:
                        self.tasks_done += 1
                        self.dark.discard(a.room)  # relit — the room is safe again
                        a.dest = None
                        self.events.append(("complete", a.id))
                else:
                    self._toward_room(a, a.dest)
            else:
                self._approach_buddy(a)  # no task to claim → buddy up for safety

        for a in self.living():
            a.trail.append((self.tick, a.room))  # each agent remembers where it has been

        self._place()  # settle cells within rooms (offline baseline; the live path stalks via follow)

        # record sightings: everyone in a room sees everyone else in it
        for room in self.rooms:
            occ = self._occ(room)
            if len(occ) > 1:
                ids = [o.id for o in occ]
                for a in occ:
                    a.seen.append(Sighting(self.tick, room, tuple(x for x in ids if x != a.id)))

        # the Cold kills: take any crewmate no one else can witness (alone in the light, or unseen in the dark)
        if self.cd == 0:
            for m in self.living(impostor=True):
                victims = [
                    c for c in self._occ(m.room) if not c.impostor and self._unwitnessed(m, c)
                ]
                if victims:
                    victim = self.rng.choice(victims)
                    m.gx, m.gy = victim.gx, victim.gy  # close in for the strike
                    victim.alive = False
                    self._release_task(victim)
                    self.bodies[m.room] = victim.id
                    self.last_kill = {"tick": self.tick, "victim": victim.id, "room": m.room}
                    self.cd = KILL_CD
                    self._flee_body(m)
                    break

        if self._check_win():
            return None

        return self._report_body()

    # --- autonomous mode: agents DECIDE via the LLM (tool calls); the engine only EXECUTES intents. ---
    # The live server reads the board from Artel, asks each free/interrupted agent for one tool call, then
    # sets the resulting intent here. step() above is kept for the offline/no-LLM path. These helpers are
    # the contract: what's a legal action right now, and how an action becomes movement/state.

    def needs_decision(self, a) -> bool:
        # an agent is at a decision point when nothing is committed — not working, walking, or following
        return a.alive and a.work == 0 and a.dest is None and a.goto is None and a.follow is None

    def legal_kills(self, m) -> list[int]:
        # crew the Cold can take THIS tick: off cooldown, past grace, within reach, and unwitnessed — alone
        # with it in a lit room, or out of sight of any crew in a dark one (see _unwitnessed). Any room now.
        if self.cd > 0 or self.tick < START_GRACE:
            return []
        return [
            c.id
            for c in self._occ(m.room)
            if not c.impostor and self._near(m, c) and self._unwitnessed(m, c)
        ]

    def prime_kill(self, a) -> bool:
        # pull the Cold into a decision whenever it shares a room with a crewmate — it will sabotage a lit
        # room, close on a victim, or strike if it already has an unwitnessed one in the dark.
        if not (a.impostor and self.cd == 0 and self.tick >= START_GRACE):
            return False
        return any(not c.impostor for c in self._occ(a.room) if c.id != a.id)

    def _flee_body(self, m) -> None:
        # the Cold never lingers over a kill: it slips away at once — by vent or on foot — and retreats
        # toward the DARK where it can strike again, not into the light beside the body. Among the dark it
        # drifts to where a lone crewmate already is (its next setup); it avoids walking into a crowd.
        dests = list(self.vents.get(m.room, ())) + sorted(self.adj.get(m.room, ()))
        if not dests:
            return
        self.rng.shuffle(dests)
        crew_n = {r: sum(1 for c in self._occ(r) if not c.impostor) for r in dests}
        m.room = max(dests, key=lambda r: (r in self.dark, crew_n[r] == 1, -crew_n[r]))

    def _avoid_bodies(self) -> None:
        # the Cold never returns to a kill: drop any intent aimed at a room holding a corpse, and if it's
        # standing in one, slip out before a crewmate walks in and catches it over the body it left.
        for m in self.living(impostor=True):
            if m.dest in self.bodies:
                m.dest = None
            if m.goto in self.bodies:
                m.goto = None
            if m.room in self.bodies:
                self._flee_body(m)

    def _release_task(self, victim) -> None:
        # a crewmate killed mid-task frees its claim: the unfinished relight goes back on the board for the
        # living to pick up, instead of the dark room staying claimed by a corpse for the rest of the game.
        if victim.dest is not None and victim.dest not in self.open_tasks:
            self.open_tasks.append(victim.dest)
        victim.dest = victim.goto = victim.follow = None
        victim.work = 0
        victim.tasking = False

    def do_kill(self, m, victim_id: int, force: bool = False) -> bool:
        occ = self._occ(m.room)
        victim = next((c for c in occ if c.id == victim_id and not c.impostor), None)
        # no kill unless the Cold is within reach of the victim and no crew witnesses it (alone in a lit
        # room, or out of sight in a dark one — see _unwitnessed). The final-hunt force-kill is exempt.
        if (
            victim is None
            or (self.cd > 0 and not force)
            or (not force and (not self._near(m, victim) or not self._unwitnessed(m, victim)))
        ):
            return False
        victim.alive = False
        self._release_task(victim)
        self.bodies[m.room] = victim.id
        self.last_kill = {"tick": self.tick, "victim": victim.id, "room": m.room}
        self.cd = KILL_CD
        self._flee_body(m)
        return True

    def _final_hunt(self) -> None:
        # the mask comes off. With one crewmate left, the Cold stops pretending — it makes straight for
        # them and takes them the moment it shares the room (no cooldown, no caution). The lone survivor
        # bolts to a room with no hunter in it, but the chase can't last: after a few ticks it's run down.
        crew = self.living(impostor=False)
        if len(crew) != 1:
            self.hunting = False
            return
        self.hunting = True
        self.hunt_ticks += 1
        victim = crew[0]
        imps = self.living(impostor=True)
        here = next((m for m in imps if m.room == victim.room), None)
        if here or self.hunt_ticks >= 6:  # cornered, or the Cold simply outruns the flee
            killer = here or imps[0]
            killer.room = victim.room
            self.do_kill(killer, victim.id, force=True)
            return
        safe = [r for r in self.adj.get(victim.room, []) if not any(m.room == r for m in imps)]
        opts = safe or self.adj.get(victim.room, [])
        if opts:
            victim.room = self.rng.choice(opts)
            victim.dest = victim.goto = victim.follow = None
        for m in imps:
            self._toward_room(m, victim.room)

    def claim_room(self, a, room: str) -> bool:
        # take a console off the board for this agent and set it walking (the Artel claim is the server's
        # job — it's the real contention arbiter; this just records the local intent + emits the event)
        if room not in self.open_tasks:
            return False
        self.open_tasks.remove(room)
        a.dest = room
        a.goto = None
        a.follow = None
        return True

    def set_goto(self, a, room: str) -> bool:
        if room not in self.adj.get(a.room, []) and room != a.room:
            return False
        a.goto = None if room == a.room else room
        a.dest = None
        a.follow = None
        return True

    def set_follow(self, a, target_id: int) -> bool:
        t = next((o for o in self.living() if o.id == target_id and o.id != a.id), None)
        if t is None:
            return False
        a.follow = target_id
        a.follow_ticks = FOLLOW_TICKS
        a.dest = None
        a.goto = None
        return True

    def default_action(self, a) -> dict:
        # the safe fallback when the LLM is unavailable or returns no/invalid tool call: claim the nearest
        # open task, else SPREAD OUT — drift to the least-crowded neighbouring room rather than mobbing a
        # buddy. Keeps the crew sweeping the station (and giving the Cold real targets) instead of blobbing.
        if self.open_tasks:
            room = min(self.open_tasks, key=lambda r: self._room_dist(a.room, r))
            return {"name": "go_to_task", "args": {"room": room}}
        nbrs = self.adj.get(a.room) or set()
        if nbrs:
            occ = {r: sum(1 for o in self.living() if o.room == r) for r in nbrs}
            tgt = min(nbrs, key=lambda r: (occ[r], self._room_dist(a.room, r)))
            return {"name": "move_to", "args": {"room": tgt}}
        return {"name": "wait", "args": {}}

    def execute(self) -> Meeting | None:
        # one tick of MECHANICS only: move each agent along its committed intent, progress tasks, record
        # sightings, then check for a body/win. No decisions and no kills are made here — those arrive as
        # tool calls the server applies before calling this. The lone-survivor final hunt is the exception:
        # once one crew remains the engine scripts the chase to its end rather than waiting on tool calls.
        self.tick += 1
        self.events = []
        if self.cd > 0:
            self.cd -= 1
        if self.living(impostor=True) and len(self.living(impostor=False)) <= 1:
            self._final_hunt()
            self._place()
            self._check_win()
            return None
        if self.sab_cd > 0:
            self.sab_cd -= 1
        if self._storm_due():
            self._storm()
        self._tick_integrity()
        for a in self.living():
            a.tasking = False
            if a.work > 0:
                a.work -= 1
                a.tasking = True
                if a.work == 0:
                    self.tasks_done += 1
                    self.dark.discard(a.room)  # relit — the room is safe again
                    a.dest = None
                    self.events.append(("complete", a.id))
            elif a.dest is not None:
                if a.room == a.dest:
                    a.work = WORK_TICKS - 1
                    a.tasking = True
                    if a.work == 0:
                        self.tasks_done += 1
                        self.dark.discard(a.room)  # relit — the room is safe again
                        a.dest = None
                        self.events.append(("complete", a.id))
                else:
                    self._toward_room(a, a.dest)
            elif a.goto is not None:
                if a.room == a.goto:
                    a.goto = None
                else:
                    self._toward_room(a, a.goto)
            elif a.follow is not None:
                a.follow_ticks -= 1
                buddy = next((o for o in self.living() if o.id == a.follow), None)
                if buddy is None or a.follow_ticks <= 0:
                    a.follow = None  # buddy gone, or time to stop and reassess
                elif buddy.room != a.room:
                    self._toward_room(a, buddy.room)
        self._avoid_bodies()  # the Cold slips out of any room holding a corpse before it can be seen there
        self._place()  # settle each agent's cell within its (now final) room — the Cold creeps into reach
        for a in self.living():
            a.trail.append((self.tick, a.room))
        for room in self.rooms:
            occ = self._occ(room)
            if len(occ) > 1:
                ids = [o.id for o in occ]
                for a in occ:
                    a.seen.append(Sighting(self.tick, room, tuple(x for x in ids if x != a.id)))
        if self._check_win():
            return None
        return self._report_body()

    def _check_win(self) -> bool:
        if not self.living(impostor=True):
            self.winner, self.win_by = "crew", "ejection"
            return True
        if not self.living(impostor=False):  # the Cold has taken the last of the crew
            self.winner, self.win_by = "impostor", "extinction"
            return True
        if self.integrity_on and self.integrity <= 0:  # the station bled out in the dark — blackout
            self.winner, self.win_by = "impostor", "blackout"
            return True
        if (
            self.storm_by_ticks and self.tick >= STORM_TICKS
        ):  # the crew outlasted the storm — dawn, and they're still standing
            self.winner, self.win_by = "crew", "storm"
            return True
        return False

    def reconvene(self) -> None:
        # a meeting was called — EVERYONE downs tools and gathers at the Mess Hall table, including any
        # crew mid-task. (Testimony — seen/trail — is kept; it's spent later in apply_votes.)
        for a in self.living():
            a.room = HUB
            a.work = 0
            a.dest = None
            a.goto = None
            a.follow = None
            a.tasking = False

    # --- meeting: collect votes via `decide`, eject the plurality, reconvene. ---
    def run_meeting(self, mt: Meeting, decide):
        self.apply_votes(mt, decide(self, mt))

    # apply an already-collected ballot (voter id -> target id, -1 = skip). The async LLM meeting
    # builds the ballot itself and calls this directly; the deterministic path routes through decide.
    def apply_votes(self, mt: Meeting, votes: dict):
        mt.votes = votes
        tally: dict[int, int] = {}
        for t in votes.values():
            tally[t] = tally.get(t, 0) + 1
        ejected = None
        if tally:
            top = max(tally.values())
            leaders = [t for t, c in tally.items() if c == top]
            if len(leaders) == 1 and leaders[0] != -1:  # tie or skip-plurality → nobody out
                ejected = leaders[0]
        if ejected is not None:
            ej = self.by_id(ejected)
            ej.alive = False
            mt.ejected = ejected
            if ej.impostor:
                self.ejected_impostors += 1
            else:
                self.wrong_ejections += 1
        self.meetings.append(mt)
        for a in self.living():
            a.room = HUB
            a.seen.clear()  # testimony is spent; co-location memory resets after the meeting
            a.trail.clear()
            a.work = 0  # drop any in-progress task; everyone reconvened at the hub
            a.dest = None
            a.goto = None
            a.follow = None
        self.cd = KILL_CD
        self._check_win()


def new_game(seed: int, n=6, impostors=1) -> Game:
    rng = random.Random(seed)
    rooms, adj, vents, rects, doors, centers, corridor = _generate_station(rng, _room_count(n))
    order = list(range(n))
    rng.shuffle(order)
    imp_ids = set(order[:impostors])
    names = rng.sample(NAMES, n)
    agents = [Agent(i, names[i], i in imp_ids, rng.choice(rooms)) for i in range(n)]
    g = Game(
        rng=rng,
        agents=agents,
        cd=START_GRACE,
        rooms=rooms,
        adj=adj,
        vents=vents,
        rects=rects,
        doors=doors,
        centers=centers,
        corridor=corridor,
        outpost=rng.randint(1, 99),
    )
    # the dark knobs are tuned at 12 rooms; scale them by station size so small stations stay playable
    g.dark_cap = round(DARK_CAP * len(g.rooms) / 12)
    g.free_dark = round(INTEGRITY_FREE_DARK * len(g.rooms) / 12)
    start_dark = min(g.dark_cap, round(START_DARK * len(g.rooms) / 12))
    g.dark = set(g.rng.sample(g.rooms, start_dark))  # a few rooms already unlit at dawn
    g.open_tasks = list(g.dark)  # the relight board starts as those dark rooms
    g._place()  # seed each agent's cell so the first snapshot already has positions
    return g


def play(seed: int, decide, n=6, impostors=1) -> Game:
    g = new_game(seed, n, impostors)
    while g.winner is None and g.tick < MAX_TICKS:
        mt = g.step()
        if mt is not None:
            g.run_meeting(mt, decide)
    if g.winner is None:
        g.winner, g.win_by = "impostor", "timeout"
    return g
