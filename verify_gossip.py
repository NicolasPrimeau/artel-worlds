from __future__ import annotations

import asyncio

from pitch.dist import GOSSIP_EVERY, DistTeam, make_dist_brain
from pitch.engine import Pitch
from pitch.gossip import ArtelGossip


class FakeArtel:
    # in-memory stand-in for Artel events: proves the shared belief is built ONLY from the read path
    def __init__(self, down: bool = False) -> None:
        self.events: list = []
        self.down = down
        self.t = 0

    async def emit_event(self, role, etype, payload):
        if self.down:
            return None
        self.t += 1
        self.events.append(
            {"created_at": f"2026-06-16T{self.t:013d}Z", "type": etype, "payload": payload}
        )
        return {}

    async def poll_events(self, role, etype, since):
        if self.down:
            return []  # Artel unreachable -> no read -> no shared belief
        return [e for e in self.events if e["created_at"] > since and e["type"] == etype]


async def play(seed: int, artel_down: bool) -> tuple:
    p = Pitch(seed=seed)
    p.setup(["x"] * 9, ["y"] * 9)
    home = DistTeam("home", bus_on=True, artel_backed=True)  # coordinates THROUGH Artel
    away = DistTeam("away", bus_on=False)  # blind control
    g = ArtelGossip(FakeArtel(down=artel_down), "home")
    g.since = "1970-01-01T00:00:00.000Z"  # the fake uses synthetic stamps; poll from the start
    brain = make_dist_brain(home, away)
    while p.tick < p.cfg.match_ticks:
        p.step(brain)
        if p.tick % GOSSIP_EVERY == 0:
            await home.gossip_cycle(p, g)
    known = len(g.shared.opp) + len(g.shared.mate) + (1 if g.shared.ball else 0)
    return p.score, known


async def main() -> None:
    for label, down in (("ARTEL UP", False), ("ARTEL DOWN", True)):
        hw = aw = dr = 0
        hg = ag = 0
        known_total = 0
        for s in range(16):
            score, known = await play(s, down)
            hg += score["home"]
            ag += score["away"]
            known_total += known
            if score["home"] > score["away"]:
                hw += 1
            elif score["away"] > score["home"]:
                aw += 1
            else:
                dr += 1
        print(
            f"{label:<12} Artel-team {hw}W {aw}L {dr}D | goals {hg / 16:.1f} vs {ag / 16:.1f} | "
            f"avg entities in shared belief (of ~17): {known_total / 16:.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
