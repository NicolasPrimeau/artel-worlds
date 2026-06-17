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
# — the Thing wears a friend's face. The Mess Hall is the hub where everyone reconvenes for a meeting.
HUB = "Mess Hall"
# The 12 rooms of the outpost. Their floorplan and connections are generated fresh every game
# (_generate_station) — a rectangle recursively sliced into abutting rooms, never the same twice — so
# no two outposts look or play alike. 12 rooms is enough that the Thing can get a victim ALONE, so most
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
    # A real floorplan on an integer tile grid: recursively slice the grid into 12 rooms (BSP) so they
    # TILE with NO gaps — rooms abut and share walls, never strung along hallways. Doors are cut where
    # two rooms share a wall; a spanning tree keeps it connected, extra doors add loops, the rest wall
    # off (chokepoints). Crawlspaces link a few far rooms. Integer cuts → exact tiling in isometric.
    m = _ROOM_MIN

    def splittable(r):
        return r[2] >= 2 * m or r[3] >= 2 * m

    leaves = [(0, 0, GW, GH)]
    while len(leaves) < 12:
        cands = sorted(
            (i for i in range(len(leaves)) if splittable(leaves[i])),
            key=lambda i: -leaves[i][2] * leaves[i][3],
        )
        if not cands:
            break
        x, y, w, h = leaves.pop(cands[rng.randrange(min(3, len(cands)))])
        if w >= 2 * m and (h < 2 * m or w >= h or rng.random() < 0.5):
            cut = rng.randint(m, w - m)
            leaves += [(x, y, cut, h), (x + cut, y, w - cut, h)]
        else:
            cut = rng.randint(m, h - m)
            leaves += [(x, y, w, cut), (x, y + cut, w, h - cut)]
    leaves.sort(key=lambda r: -r[2] * r[3])
    rects = {SIZE_ORDER[i]: tuple(leaves[i]) for i in range(12)}
    names = list(rects)

    def shared(ra, rb):  # (overlap_len, (door_x, door_y)) if the rects abut along a wall, else None
        ax, ay, aw, ah = ra
        bx, by, bw, bh = rb
        if ax + aw == bx or bx + bw == ax:
            lo, hi = max(ay, by), min(ay + ah, by + bh)
            if hi - lo >= 1:
                return (hi - lo, (ax + aw if ax + aw == bx else bx + bw, (lo + hi) / 2))
        if ay + ah == by or by + bh == ay:
            lo, hi = max(ax, bx), min(ax + aw, bx + bw)
            if hi - lo >= 1:
                return (hi - lo, ((lo + hi) / 2, ay + ah if ay + ah == by else by + bh))
        return None

    walls = {}  # (a,b) -> (overlap, door point) for every pair that physically shares a wall
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            s = shared(rects[a], rects[b])
            if s:
                walls[(a, b)] = s
    nbr: dict[str, list] = {n: [] for n in names}
    for a, b in walls:
        nbr[a].append(b)
        nbr[b].append(a)
    adj: dict[str, set] = {n: set() for n in names}
    doors: dict = {}
    seen, q = {names[0]}, [names[0]]  # spanning tree over ALL shared walls → always connected
    while q:
        u = q.pop(0)
        for v in nbr[u]:
            if v not in seen:
                seen.add(v)
                adj[u].add(v)
                adj[v].add(u)
                doors[(u, v)] = (walls.get((u, v)) or walls[(v, u)])[1]
                q.append(v)
    for (a, b), (
        ov,
        pt,
    ) in walls.items():  # extra doors on the longer walls → loops; rest stay walled
        if b not in adj[a] and ov >= _MIN_DOOR and rng.random() < 0.5:
            adj[a].add(b)
            adj[b].add(a)
            doors[(a, b)] = pt
    vents: dict[str, list] = {}
    nonadj = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if b not in adj[a]]
    rng.shuffle(nonadj)
    for a, b in nonadj[: rng.randint(2, 3)]:
        vents.setdefault(a, []).append(b)
        vents.setdefault(b, []).append(a)
    return names, {k: sorted(v) for k, v in adj.items()}, vents, rects, doors


TASKS_EACH = 5
TASK_P = 0.22  # per-tick chance a crew STARTS a task (then it stays put working for WORK_TICKS)
WORK_TICKS = 3  # a task occupies its crew for several ticks — they linger at the console
KILL_CD = 6  # ticks between kills — on a 12-room map 2 Things are a real threat; 1 can't cover it
START_GRACE = 6  # no kill on the first few ticks, so a real task phase builds movement + alibis
EMERGENCY_P = 0.01  # per-tick chance a crew calls a meeting on suspicion alone
MAX_TICKS = 600

# the winter-over crew — AI/ML pun surnames (the joke: a fleet of language models playing The Thing,
# named for the machinery that runs them). Each game samples a distinct subset from the seed, so names
# overlap between games but never repeat within one (pitch-style).
NAMES = [
    "Softmaxwell",  # softmax
    "Overfitz",  # overfit
    "Hallucinov",  # hallucinate
    "Frostbyte",  # byte (and it's the Antarctic)
    "Beamsworth",  # beam search
    "ReLuther",  # ReLU
    "Sigmund",  # sigmoid
    "Dropoutski",  # dropout
    "Attenborough",  # attention
    "Embeddington",  # embedding
    "Perplexton",  # perplexity
    "Quantz",  # quantize
    "Gradiev",  # gradient
    "Tensorova",  # tensor
    "Lossman",  # loss
    "Adamson",  # Adam optimizer
    "Batchelor",  # batch norm
    "Tokarev",  # token
    "Logitsky",  # logits
    "Inferenza",  # inference
    "Vectorov",  # vector
    "Kernighan",  # kernel
    "Bayesworth",  # Bayes
    "Markova",  # Markov chain
    "Boltzmann",  # Boltzmann machine
    "Entropov",  # entropy
    "Cudahy",  # CUDA
    "Pruitt",  # pruning
    "Temperton",  # temperature
    "Distilla",  # distillation
    "Epochwell",  # epoch
    "Convoluto",  # convolution
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
    rects: dict = field(default_factory=dict)  # name -> (x,y,w,h) floorplan rectangle
    doors: dict = field(default_factory=dict)  # (a,b) -> (x,y) doorway point on the shared wall
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

    def living(self, impostor=None):
        return [a for a in self.agents if a.alive and (impostor is None or a.impostor == impostor)]

    def by_id(self, i):
        return next(a for a in self.agents if a.id == i)

    def _occ(self, room):
        return [a for a in self.living() if a.room == room]

    def tasks_left(self):
        return sum(a.tasks for a in self.living(impostor=False))

    # --- task phase: one tick of move / task / kill. Returns a Meeting trigger or None. ---
    def step(self) -> Meeting | None:
        self.tick += 1
        if self.cd > 0:
            self.cd -= 1

        for a in self.living():
            a.tasking = False
            if a.impostor:
                if a.room in self.vents and self.rng.random() < 0.25:
                    a.room = self.rng.choice(self.vents[a.room])  # vent away secretly
                else:
                    a.room = self.rng.choice(self.adj[a.room])
            elif a.work > 0:  # CLAIMED a task: stay at the console, working until done
                a.work -= 1
                a.tasking = True
                if a.work == 0:
                    a.tasks -= 1  # COMPLETE — one task off the shared Artel board, then move on
            elif a.tasks > 0 and self.rng.random() < TASK_P:
                a.work = WORK_TICKS - 1  # claim a task — occupies this tick + the next few
                a.tasking = True
                if a.work == 0:
                    a.tasks -= 1
            else:
                a.room = self.rng.choice(self.adj[a.room])

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
                if isolated or self.rng.random() < 0.4:
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
        self.cd = KILL_CD
        self._check_win()


def new_game(seed: int, n=6, impostors=1) -> Game:
    rng = random.Random(seed)
    rooms, adj, vents, rects, doors = _generate_station(rng)
    order = list(range(n))
    rng.shuffle(order)
    imp_ids = set(order[:impostors])
    names = rng.sample(NAMES, n)
    agents = [Agent(i, names[i], i in imp_ids, rng.choice(rooms)) for i in range(n)]
    return Game(
        rng=rng,
        agents=agents,
        cd=START_GRACE,
        rooms=rooms,
        adj=adj,
        vents=vents,
        rects=rects,
        doors=doors,
        outpost=rng.randint(1, 99),
    )


def play(seed: int, decide, n=6, impostors=1) -> Game:
    g = new_game(seed, n, impostors)
    while g.winner is None and g.tick < MAX_TICKS:
        mt = g.step()
        if mt is not None:
            g.run_meeting(mt, decide)
    if g.winner is None:
        g.winner, g.win_by = "impostor", "timeout"
    return g
