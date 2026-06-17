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
    # An ORGANIC cluster, not a grid: scatter 12 seeds inside a blob (an ellipse) and grow them by
    # multi-source flood fill, so each room is an IRREGULAR region of tiles that abuts its neighbours.
    # The silhouette is a rough blob; rooms are non-rectangular and no two are alike. Doors sit on shared
    # borders (spanning tree keeps it connected, extra doors add loops). Crawlspaces link a few far rooms.
    import math
    from collections import defaultdict, deque

    cx, cy, rx, ry = GW / 2, GH / 2, GW * 0.48, GH * 0.46
    inmask = [
        [((x + 0.5 - cx) / rx) ** 2 + ((y + 0.5 - cy) / ry) ** 2 <= 1.0 for x in range(GW)]
        for y in range(GH)
    ]
    mask = [(x, y) for y in range(GH) for x in range(GW) if inmask[y][x]]
    seeds, mind = [], math.sqrt(len(mask) / 12) * 0.82  # 12 spread-out seeds
    pool = mask[:]
    rng.shuffle(pool)
    for t in pool:
        if all((t[0] - s[0]) ** 2 + (t[1] - s[1]) ** 2 >= mind * mind for s in seeds):
            seeds.append(t)
            if len(seeds) == 12:
                break
    for t in pool:  # relax if min-distance left us short
        if len(seeds) == 12:
            break
        if t not in seeds:
            seeds.append(t)
    owner = [[-1] * GW for _ in range(GH)]
    q = deque()
    for i, (s0, s1) in enumerate(seeds):
        owner[s1][s0] = i
        q.append((s0, s1))
    while q:  # multi-source flood → 12 contiguous, irregular regions filling the blob
        x, y = q.popleft()
        o = owner[y][x]
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < GW and 0 <= ny < GH and inmask[ny][nx] and owner[ny][nx] == -1:
                owner[ny][nx] = o
                q.append((nx, ny))
    sizes = [0] * 12
    for row in owner:
        for o in row:
            if o >= 0:
                sizes[o] += 1
    by_size = sorted(range(12), key=lambda i: -sizes[i])  # biggest region -> Mess Hall, etc.
    idx2name = {by_size[i]: SIZE_ORDER[i] for i in range(12)}
    names = list(SIZE_ORDER)
    grid = [
        [idx2name[owner[y][x]] if owner[y][x] >= 0 else None for x in range(GW)] for y in range(GH)
    ]
    sx, sy, cnt = defaultdict(float), defaultdict(float), defaultdict(int)
    walls = defaultdict(list)  # (a,b) sorted -> shared-border door points
    for y in range(GH):
        for x in range(GW):
            n = grid[y][x]
            if not n:
                continue
            sx[n] += x + 0.5
            sy[n] += y + 0.5
            cnt[n] += 1
            for nx, ny, pt in ((x + 1, y, (x + 1, y + 0.5)), (x, y + 1, (x + 0.5, y + 1))):
                if 0 <= nx < GW and 0 <= ny < GH and grid[ny][nx] and grid[ny][nx] != n:
                    walls[tuple(sorted((n, grid[ny][nx])))].append(pt)
    centers = {n: (round(sx[n] / cnt[n], 2), round(sy[n] / cnt[n], 2)) for n in names}
    nbr: dict[str, list] = defaultdict(list)
    for a, b in walls:
        nbr[a].append(b)
        nbr[b].append(a)
    adj: dict[str, set] = {n: set() for n in names}
    doors: dict = {}

    def mid(pts):
        return sorted(pts)[len(pts) // 2]

    seen, bq = {names[0]}, deque([names[0]])  # spanning tree over shared borders → connected
    while bq:
        u = bq.popleft()
        for v in nbr[u]:
            if v not in seen:
                seen.add(v)
                adj[u].add(v)
                adj[v].add(u)
                doors[(u, v)] = mid(walls[tuple(sorted((u, v)))])
                bq.append(v)
    for (a, b), pts in walls.items():  # extra doors on longer borders -> loops; rest stay walled
        if b not in adj[a] and len(pts) >= _MIN_DOOR and rng.random() < 0.5:
            adj[a].add(b)
            adj[b].add(a)
            doors[(a, b)] = mid(pts)
    vents: dict[str, list] = {}
    nonadj = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if b not in adj[a]]
    rng.shuffle(nonadj)
    for a, b in nonadj[: rng.randint(2, 3)]:
        vents.setdefault(a, []).append(b)
        vents.setdefault(b, []).append(a)
    return names, {k: sorted(v) for k, v in adj.items()}, vents, grid, doors, centers


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
    grid: list = field(
        default_factory=list
    )  # GH x GW tiles of room-name (or None) — organic regions
    doors: dict = field(default_factory=dict)  # (a,b) -> (x,y) doorway point on the shared border
    centers: dict = field(default_factory=dict)  # name -> (cx,cy) region centroid in tiles
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
    rooms, adj, vents, grid, doors, centers = _generate_station(rng)
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
        grid=grid,
        doors=doors,
        centers=centers,
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
