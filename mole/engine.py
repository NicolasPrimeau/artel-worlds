from __future__ import annotations

import random
from dataclasses import dataclass, field

# A social-deduction world (Among Us / Werewolf shape). The deterministic engine runs the visible
# loop — agents move room to room, crew do tasks, the MOLE kills when it can, a found body forces a
# meeting and a vote. What an agent KNOWS is partial: it only sees who shares its room. The whole
# game turns on whether agents can POOL what they saw — that pooling is the load-bearing job (Artel
# live; here a toggle for the A/B). Communication isn't a tactic here, it's the win condition.

ROOMS = ["Cafeteria", "Medbay", "Electrical", "Storage", "Reactor", "Navigation"]
ADJ = {
    "Cafeteria": ["Medbay", "Storage", "Navigation"],
    "Medbay": ["Cafeteria", "Electrical"],
    "Electrical": ["Medbay", "Storage"],
    "Storage": ["Electrical", "Cafeteria", "Reactor"],
    "Reactor": ["Storage", "Navigation"],
    "Navigation": ["Reactor", "Cafeteria"],
}
TASKS_EACH = 3
KILL_CD = 8  # ticks between kills
MAX_TICKS = 400


@dataclass
class Agent:
    id: int
    name: str
    mole: bool
    room: str
    alive: bool = True
    tasks: int = TASKS_EACH
    knows: set = field(default_factory=set)  # mole ids this agent witnessed killing


@dataclass
class Game:
    rng: random.Random
    agents: list
    bodies: dict = field(default_factory=dict)  # room -> victim id, undiscovered
    tick: int = 0
    cd: int = 0
    winner: str | None = None  # "crew" | "mole"
    meetings: int = 0
    ejected_moles: int = 0
    wrong_ejections: int = 0

    def alive(self, mole=None):
        return [a for a in self.agents if a.alive and (mole is None or a.mole == mole)]

    def _occ(self, room):
        return [a for a in self.alive() if a.room == room]

    # --- the task phase: move, do tasks, kill; returns True if a body triggers a meeting ---
    def step(self) -> bool:
        self.tick += 1
        if self.cd > 0:
            self.cd -= 1
        for a in self.alive():
            if self.rng.random() < 0.6:  # wander
                a.room = self.rng.choice(ADJ[a.room])
            elif not a.mole and a.tasks > 0:  # stay and work a task
                a.tasks -= 1
        # the mole kills when off cooldown and alone-ish with crew; bystanders WITNESS it
        for m in self.alive(mole=True):
            if self.cd > 0:
                continue
            crew_here = [c for c in self._occ(m.room) if not c.mole and c.id != m.id]
            if crew_here and self.rng.random() < 0.7:
                victim = self.rng.choice(crew_here)
                victim.alive = False
                self.bodies[m.room] = victim.id
                for w in self._occ(m.room):  # whoever else is in the room saw it
                    if w.id != m.id:
                        w.knows.add(m.id)
                self.cd = KILL_CD
                break
        # crew win ONLY by ejecting every mole (deduction is the whole game); tasks are flavour
        if len(self.alive(mole=True)) >= len(self.alive(mole=False)):
            self.winner = "mole"
        # a body is found when a living soul shares its room
        for room, _victim in list(self.bodies.items()):
            if any(a.room == room for a in self.alive()):
                del self.bodies[room]
                return True
        return False

    def _crew_tasks_left(self):
        return sum(a.tasks for a in self.alive(mole=False))

    # --- the meeting: accuse + vote. `share` = can agents pool what they saw (the Artel job)? ---
    def meeting(self, share: bool):
        self.meetings += 1
        voters = self.alive()
        n = len(voters)
        # eyewitnesses (crew who were in the room when the mole struck) each name the mole they saw
        witness_votes: dict[int, int] = {}
        for v in voters:
            if not v.mole and v.knows:
                m = next(iter(v.knows))
                witness_votes[m] = witness_votes.get(m, 0) + 1

        ejected = None
        if share:
            # testimony is POOLED: a credible eyewitness account, told to the whole room, convicts;
            # corroboration just breaks ties. The mole's baseless counter-accusation is dismissed
            # because the table can compare notes and see only the witness's story is backed by a body.
            if witness_votes:
                ejected = max(witness_votes, key=witness_votes.get)
        else:
            # SILOED: a witness is one vote in a sea of "I didn't see anything", and the mole casts a
            # deflection of its own. Eject only on a strict majority of the table — which a witness or
            # two scattered among skips almost never reaches, so the mole walks free.
            tally = dict(witness_votes)
            for v in voters:
                if v.mole:
                    targets = [c.id for c in voters if not c.mole]
                    if targets:
                        t = self.rng.choice(targets)
                        tally[t] = tally.get(t, 0) + 1
            if tally:
                top = max(tally, key=tally.get)
                if tally[top] > n // 2:
                    ejected = top

        if ejected is not None:
            ej = next(a for a in self.agents if a.id == ejected)
            ej.alive = False
            if ej.mole:
                self.ejected_moles += 1
            else:
                self.wrong_ejections += 1
        for a in self.alive():
            a.room = "Cafeteria"  # reconvene; cooldown resets (Among Us-style)
        self.cd = KILL_CD
        if not self.alive(mole=True):
            self.winner = "crew"
        elif len(self.alive(mole=True)) >= len(self.alive(mole=False)):
            self.winner = "mole"


def new_game(seed: int, n=6, moles=1) -> Game:
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    mole_ids = set(order[:moles])
    agents = [Agent(i, f"P{i}", i in mole_ids, rng.choice(ROOMS)) for i in range(n)]
    return Game(rng=rng, agents=agents)


def play(seed: int, share: bool, n=6, moles=1) -> Game:
    g = new_game(seed, n, moles)
    while g.winner is None and g.tick < MAX_TICKS:
        if g.step():
            g.meeting(share)
    if g.winner is None:
        g.winner = "mole"  # never caught — the mole evaded to the time limit
    return g
