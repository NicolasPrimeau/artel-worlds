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

ROOMS = ["Cafeteria", "Medbay", "Electrical", "Storage", "Reactor", "Navigation"]
ADJ = {
    "Cafeteria": ["Medbay", "Storage", "Navigation"],
    "Medbay": ["Cafeteria", "Electrical"],
    "Electrical": ["Medbay", "Storage"],
    "Storage": ["Electrical", "Cafeteria", "Reactor"],
    "Reactor": ["Storage", "Navigation"],
    "Navigation": ["Reactor", "Cafeteria"],
}
# vents let the impostor move secretly between a few far-apart rooms (escape a fresh body)
VENTS = {
    "Electrical": ["Reactor"],
    "Reactor": ["Electrical"],
    "Medbay": ["Navigation"],
    "Navigation": ["Medbay"],
}
TASKS_EACH = 5
TASK_P = 0.18  # per-tick chance a crew advances a task (the board is a slow race, not a sprint)
KILL_CD = 5  # ticks between kills — fast enough to threaten before the task board clears
EMERGENCY_P = 0.01  # per-tick chance a crew calls a meeting on suspicion alone
MAX_TICKS = 600

COLORS = ["Red", "Blue", "Green", "Pink", "Orange", "Yellow", "Black", "White", "Cyan", "Lime"]


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
    # what this agent privately observed — the raw material for its testimony in a meeting
    seen: list = field(default_factory=list)  # list[Sighting]
    witnessed: set = field(default_factory=set)  # impostor ids it directly saw kill
    found: list = field(default_factory=list)  # (tick, room, victim_id) bodies it discovered


@dataclass
class Meeting:
    tick: int
    reporter: int  # who called it (-1 = emergency button)
    room: str  # where the body was (or "Cafeteria" for emergency)
    victim: int | None  # who died, if a body report
    ejected: int | None = None  # who got voted out
    votes: dict = field(default_factory=dict)  # voter id -> target id (or -1 for skip)


@dataclass
class Game:
    rng: random.Random
    agents: list
    bodies: dict = field(default_factory=dict)  # room -> victim id, undiscovered
    tick: int = 0
    cd: int = 0  # shared impostor kill cooldown
    winner: str | None = None  # "crew" | "impostor"
    win_by: str | None = None  # "tasks" | "ejection" | "parity" | "timeout"
    meetings: list = field(default_factory=list)
    ejected_impostors: int = 0
    wrong_ejections: int = 0

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
            if a.impostor:
                if a.room in VENTS and self.rng.random() < 0.25:
                    a.room = self.rng.choice(VENTS[a.room])  # vent away secretly
                else:
                    a.room = self.rng.choice(ADJ[a.room])
            elif a.tasks > 0 and self.rng.random() < TASK_P:
                a.tasks -= 1  # claim + complete one task on the shared board
            else:
                a.room = self.rng.choice(ADJ[a.room])

        # record sightings: everyone in a room sees everyone else in it
        for room in ROOMS:
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
                    for w in self._occ(m.room):
                        if w.id != m.id:
                            w.witnessed.add(m.id)  # any survivor in the room made the impostor
                    self.cd = KILL_CD
                    if m.room in VENTS and self.rng.random() < 0.6:
                        m.room = self.rng.choice(VENTS[m.room])  # vent off the body
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
            return Meeting(self.tick, caller.id, "Cafeteria", None)

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
        votes = decide(self, mt)  # voter id -> target id (-1 = skip)
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
            a.room = "Cafeteria"
            a.seen.clear()  # testimony is spent; co-location memory resets after the meeting
        self.cd = KILL_CD
        self._check_win()


def new_game(seed: int, n=6, impostors=1) -> Game:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    imp_ids = set(order[:impostors])
    names = rng.sample(COLORS, n)
    agents = [Agent(i, names[i], i in imp_ids, rng.choice(ROOMS)) for i in range(n)]
    return Game(rng=rng, agents=agents)


def play(seed: int, decide, n=6, impostors=1) -> Game:
    g = new_game(seed, n, impostors)
    while g.winner is None and g.tick < MAX_TICKS:
        mt = g.step()
        if mt is not None:
            g.run_meeting(mt, decide)
    if g.winner is None:
        g.winner, g.win_by = "impostor", "timeout"
    return g
