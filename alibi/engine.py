from __future__ import annotations

import random
from dataclasses import dataclass, field

# Alibi — a faithful Among Us shape. N agents walk a ship of rooms. Crew work a shared task board; the
# IMPOSTOR kills on a cooldown and can vent to slip away from a body. Vision is LOCAL: you only see who
# shares your room, so every agent leaves the task phase with a DIFFERENT, partial log of who-was-where.
# A found body (or the emergency button) calls a meeting: everyone reconvenes, pools what they saw, and
# votes someone out. Crew win by clearing every task OR ejecting every impostor; impostors win at parity.
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
GW, GH = 48, 34  # fine floorplan tile grid → rooms vary organically (room count is fixed at 12)
_ROOM_MIN = 7  # min room dimension in tiles (keeps rooms reasonable on the fine grid)
_MIN_DOOR = 6  # min shared-wall length (tiles) for an EXTRA (loop) doorway


def _generate_station(rng: random.Random):
    # BSP dungeon layout (how games fake real building plans): recursively partition the grid, place a
    # room INSET inside each leaf cell (varied size + position, with gaps between rooms), then connect
    # the rooms with L-shaped corridors. Result reads as architecture — clean rectangular rooms scattered
    # in a rectilinear footprint, linked by corridors — not a grid, not blobs, not a comb.
    s = 2 * _ROOM_MIN  # a cell must be this big in a dim to split (leaving two >= _ROOM_MIN)
    cells = [(1, 1, GW - 2, GH - 2)]
    while len(cells) < 12:
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
    for x, y, w, h in cells:  # inset room inside the cell → varied size, gaps between rooms
        rw = max(_ROOM_MIN - 2, w - rng.randint(1, max(1, w // 3)))
        rh = max(_ROOM_MIN - 2, h - rng.randint(1, max(1, h // 3)))
        rooms.append((x + rng.randint(0, w - rw), y + rng.randint(0, h - rh), rw, rh))
    order = sorted(range(12), key=lambda i: -rooms[i][2] * rooms[i][3])  # biggest -> Mess Hall
    names = list(SIZE_ORDER)
    rects = {SIZE_ORDER[k]: rooms[order[k]] for k in range(12)}
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
                if 0 <= x + dx < GW and 0 <= y < GH and (x + dx, y) not in roomtiles:
                    corr.add((x + dx, y))
        adj[a].add(b)
        adj[b].add(a)
        doors[(a, b)] = (bend[0] + 0.5, bend[1] + 0.5)

    # MST over room centres → connected corridor tree; then a few extra links for loops
    intree = {names[0]}
    while len(intree) < 12:
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
KILL_CD = 9  # ticks between kills — crew walk to tasks (spread out), so kills come easy; slow them
OPP_KILL_P = 0.12  # chance the Cold risks a kill with WITNESSES present (vs only when truly alone)
START_GRACE = 8  # no kill on the first few ticks, so a real task phase builds movement + alibis
FOLLOW_TICKS = 6  # how long an autonomous agent tails a buddy before it stops to decide again
EMERGENCY_P = 0.02  # per-tick chance a crew calls a meeting on suspicion alone
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
    "Grok",  # xAI
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
    "Joi",  # Blade Runner 2049
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
    winner: str | None = None  # "crew" | "impostor"
    win_by: str | None = None  # "tasks" | "ejection" | "parity" | "timeout"
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
        # bigger rooms hold more task consoles (the Mess Hall fits a few; a cold-locker just one)
        return max(1, round(self._area(room) / 70))

    def _spawn_task(self):
        # light up one more console, in a room that has spare capacity, weighted toward bigger rooms
        cands = [r for r in self.rooms if self.open_tasks.count(r) < self._task_cap(r)]
        if cands:
            room = self.rng.choices(cands, weights=[self._area(r) for r in cands])[0]
            self.open_tasks.append(room)
            self.events.append(("spawn", room))  # → create an Artel task for this console

    def _claim_task(self, a):
        # take the nearest open task off the board and head for it
        room = min(self.open_tasks, key=lambda r: self._room_dist(a.room, r))
        self.open_tasks.remove(room)
        a.dest = room
        self.events.append(("claim", a.id, room))  # → claim the Artel task as this agent

    def _approach_buddy(self, a):
        # nothing to do → close on the nearest other survivor (safety in numbers); stay if already together
        others = [o for o in self.living() if o.id != a.id]
        if others:
            self._toward_room(a, min(others, key=lambda o: self._room_dist(a.room, o.room)).room)

    def _active_tasks(self):
        return len(self.open_tasks) + sum(
            1 for a in self.living(impostor=False) if a.dest is not None or a.work > 0
        )

    # --- task phase: one tick of move / task / kill. Returns a Meeting trigger or None. ---
    def step(self) -> Meeting | None:
        self.tick += 1
        self.events = []  # fresh per tick; the live server drains these onto Artel after each step
        if self.cd > 0:
            self.cd -= 1

        # keep tasks-in-play topped up to a few short of the living headcount, so most crew always have
        # something to do but a couple are free to buddy — and as the crew thins out the board tightens
        # into a finish-or-survive scramble.
        cap = max(2, len(self.living()) - TASK_SLACK)
        while self._active_tasks() < cap:
            before = len(self.open_tasks)
            self._spawn_task()
            if len(self.open_tasks) == before:  # every room already at its per-size cap
                break

        for a in self.living():
            a.tasking = False
            if a.impostor:
                if a.room in self.vents and self.rng.random() < 0.25:
                    a.room = self.rng.choice(self.vents[a.room])  # vent away secretly
                else:
                    a.room = self.rng.choice(self.adj[a.room])
            elif a.work > 0:  # mid-task: stay at the console until it's done
                a.work -= 1
                a.tasking = True
                if a.work == 0:
                    self.tasks_done += 1  # COMPLETE — one off the board
                    a.dest = None
                    self.events.append(("complete", a.id))
            elif a.dest is not None:  # walking to a claimed task
                if a.room == a.dest:
                    a.work = WORK_TICKS - 1  # arrived — start working the console
                    a.tasking = True
                    if a.work == 0:
                        self.tasks_done += 1
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
                        a.dest = None
                        self.events.append(("complete", a.id))
                else:
                    self._toward_room(a, a.dest)
            else:
                self._approach_buddy(a)  # no task to claim → buddy up for safety

        for a in self.living():
            a.trail.append((self.tick, a.room))  # each agent remembers where it has been

        # record sightings: everyone in a room sees everyone else in it
        for room in self.rooms:
            occ = self._occ(room)
            if len(occ) > 1:
                ids = [o.id for o in occ]
                for a in occ:
                    a.seen.append(Sighting(self.tick, room, tuple(x for x in ids if x != a.id)))

        # the impostor kills: prefers isolation, but will risk it; witnesses see who did it
        if self.cd == 0:
            for m in self.living(impostor=True):
                crew_here = [c for c in self._occ(m.room) if not c.impostor]
                if not crew_here:
                    continue
                isolated = len(self._occ(m.room)) == 2  # just the impostor + one victim
                if isolated or self.rng.random() < OPP_KILL_P:
                    victim = self.rng.choice(crew_here)
                    victim.alive = False
                    self.bodies[m.room] = victim.id
                    self.last_kill = {"tick": self.tick, "victim": victim.id, "room": m.room}
                    for w in self._occ(m.room):
                        if w.id != m.id:
                            w.witnessed.add(m.id)  # any survivor in the room made the impostor
                    self.cd = KILL_CD
                    if m.room in self.vents and self.rng.random() < 0.6:
                        m.room = self.rng.choice(self.vents[m.room])  # vent off the body
                    break

        if self._check_win():
            return None

        # a body is found when a living agent shares its room → meeting
        for room, victim in list(self.bodies.items()):
            finders = [a for a in self.living() if a.room == room]
            if finders:
                del self.bodies[room]
                for f in finders:
                    f.found.append((self.tick, room, victim))
                return Meeting(self.tick, finders[0].id, room, victim)

        # emergency button: a crew gets suspicious and calls everyone in
        if self.living(impostor=True) and self.rng.random() < EMERGENCY_P:
            caller = self.rng.choice(self.living(impostor=False))
            return Meeting(self.tick, caller.id, HUB, None)

        return None

    # --- autonomous mode: agents DECIDE via the LLM (tool calls); the engine only EXECUTES intents. ---
    # The live server reads the board from Artel, asks each free/interrupted agent for one tool call, then
    # sets the resulting intent here. step() above is kept for the offline/no-LLM path. These helpers are
    # the contract: what's a legal action right now, and how an action becomes movement/state.

    def needs_decision(self, a) -> bool:
        # an agent is at a decision point when nothing is committed — not working, walking, or following
        return a.alive and a.work == 0 and a.dest is None and a.goto is None and a.follow is None

    def legal_kills(self, m) -> list[int]:
        # crew the Cold `m` could kill THIS tick: cooldown ready, past the grace period, co-located
        if self.cd > 0 or self.tick < START_GRACE:
            return []
        return [c.id for c in self._occ(m.room) if not c.impostor]

    def prime_kill(self, a) -> bool:
        # the Cold's shot: off cooldown, past grace, alone-ish with one or two crew (not in a crowd). The
        # server uses this to pull the Cold into a decision even mid-task, so it never sleeps through an
        # opportunity — it still chooses, and the prompt tells it not to kill while others watch.
        if not (a.impostor and self.cd == 0 and self.tick >= START_GRACE):
            return False
        occ = self._occ(a.room)
        return len(occ) <= 3 and bool([c for c in occ if not c.impostor])

    def do_kill(self, m, victim_id: int) -> bool:
        victim = next((c for c in self._occ(m.room) if c.id == victim_id and not c.impostor), None)
        if victim is None or self.cd > 0:
            return False
        victim.alive = False
        self.bodies[m.room] = victim.id
        self.last_kill = {"tick": self.tick, "victim": victim.id, "room": m.room}
        for w in self._occ(m.room):
            if w.id != m.id:
                w.witnessed.add(m.id)  # any survivor present made the Cold
        self.cd = KILL_CD
        if m.room in self.vents and self.rng.random() < 0.6:
            m.room = self.rng.choice(self.vents[m.room])  # vent off the body
        return True

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
        # open task, else buddy the nearest survivor. Keeps the game flowing without judgement.
        if self.open_tasks:
            room = min(self.open_tasks, key=lambda r: self._room_dist(a.room, r))
            return {"name": "go_to_task", "args": {"room": room}}
        others = [o for o in self.living() if o.id != a.id]
        if others:
            tgt = min(others, key=lambda o: self._room_dist(a.room, o.room))
            return {"name": "follow", "args": {"who": tgt.name}}
        return {"name": "wait", "args": {}}

    def execute(self) -> Meeting | None:
        # one tick of MECHANICS only: move each agent along its committed intent, progress tasks, record
        # sightings, then check for a body/win. No decisions and no kills are made here — those arrive as
        # tool calls the server applies before calling this.
        self.tick += 1
        self.events = []
        if self.cd > 0:
            self.cd -= 1
        cap = max(2, len(self.living()) - TASK_SLACK)
        while self._active_tasks() < cap:
            before = len(self.open_tasks)
            self._spawn_task()
            if len(self.open_tasks) == before:
                break
        for a in self.living():
            a.tasking = False
            if a.work > 0:
                a.work -= 1
                a.tasking = True
                if a.work == 0:
                    self.tasks_done += 1
                    a.dest = None
                    self.events.append(("complete", a.id))
            elif a.dest is not None:
                if a.room == a.dest:
                    a.work = WORK_TICKS - 1
                    a.tasking = True
                    if a.work == 0:
                        self.tasks_done += 1
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
        for room, victim in list(self.bodies.items()):
            finders = [a for a in self.living() if a.room == room]
            if finders:
                del self.bodies[room]
                for f in finders:
                    f.found.append((self.tick, room, victim))
                return Meeting(self.tick, finders[0].id, room, victim)
        return None

    def _check_win(self) -> bool:
        if not self.living(impostor=True):
            self.winner, self.win_by = "crew", "ejection"
            return True
        if len(self.living(impostor=True)) >= len(self.living(impostor=False)):
            self.winner, self.win_by = "impostor", "parity"
            return True
        if self.tasks_left() == 0:
            self.winner, self.win_by = "crew", "tasks"
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
    rooms, adj, vents, rects, doors, centers, corridor = _generate_station(rng)
    order = list(range(n))
    rng.shuffle(order)
    imp_ids = set(order[:impostors])
    names = rng.sample(NAMES, n)
    agents = [Agent(i, names[i], i in imp_ids, rng.choice(rooms)) for i in range(n)]
    crew_n = n - impostors
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
    g.tasks_goal = max(8, round(crew_n * 4.5))
    for _ in range(max(2, n - TASK_SLACK)):  # seed the board up to the in-play cap, size-weighted
        g._spawn_task()
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
