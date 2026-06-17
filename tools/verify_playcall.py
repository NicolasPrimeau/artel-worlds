from __future__ import annotations

# Prove the play-call conduit is load-bearing: the brain publishes a call to Artel, the actuator
# reads it back from Artel. Artel UP -> calls arrive -> plays fire. Artel DOWN -> no call -> baseline.

import asyncio

from pitch.engine import Pitch
from pitch.plays import PlayManager, decide_call, play_brain, publish_call


class FakeArtel:
    def __init__(self, down=False):
        self.events = []
        self.down = down
        self.t = 0

    async def emit_event(self, role, etype, payload):
        if self.down:
            return None
        self.t += 1
        self.events.append({"created_at": f"2026-06-17T{self.t:013d}Z", "type": etype, "payload": payload})
        return {}

    async def poll_events(self, role, etype, since):
        if self.down:
            return []
        return [e for e in self.events if e["created_at"] > since and e["type"] == etype]


async def play(seed, down):
    p = Pitch(seed=seed)
    p.setup(["x"] * 9, ["y"] * 9)
    mgr = PlayManager("home")
    mgr._call_cursor = "1970-01-01T00:00:00.000Z"  # the fake uses synthetic stamps
    brain = play_brain("home", mgr)
    artel = FakeArtel(down=down)
    while p.tick < p.cfg.match_ticks:
        p.step(brain)
        if p.tick % 12 == 0:  # ~1s cadence: brain decides + publishes, actuator pulls
            await publish_call(artel, "home", decide_call(p, "home"))
            await mgr.pull_call(artel, now=p.tick * 0.08)
    return mgr.started, mgr.completed


async def main():
    for label, down in (("ARTEL UP", False), ("ARTEL DOWN", True)):
        tot_s = tot_c = 0
        for s in range(8):
            st, c = await play(s, down)
            tot_s += st
            tot_c += c
        print(f"{label:<11} plays started {tot_s}, completed {tot_c}  (over 8 matches)")


if __name__ == "__main__":
    asyncio.run(main())
