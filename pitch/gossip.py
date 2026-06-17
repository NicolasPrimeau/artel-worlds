from __future__ import annotations

from .dist import STALE, Belief

# The Artel gossip transport. A team's shared belief is assembled ONLY from events polled back out
# of Artel: each line agent (def / mid / fwd) emits what its players currently see as a `pitch.sighting`
# event; the team then polls those events and merges them into a running shared belief. There is no
# local shortcut — if Artel is unreachable the poll returns nothing, the shared belief never fills,
# and the team is left with each node's own senses only (i.e. blind). That is what makes the bus
# genuinely load-bearing rather than a mirror: cut Artel and the coordinated team becomes the blind one.

EVENT_TYPE = "pitch.sighting"
EPOCH = "1970-01-01T00:00:00.000Z"


def _ser(b: Belief) -> dict:
    return {
        "ball": list(b.ball) if b.ball else None,
        "opp": {str(k): list(v) for k, v in b.opp.items()},
        "mate": {str(k): list(v) for k, v in b.mate.items()},
    }


def _merge_into(shared: Belief, p: dict) -> None:
    bl = p.get("ball")
    if bl and (shared.ball is None or bl[2] > shared.ball[2]):
        shared.ball = tuple(bl)
    for k, v in p.get("opp", {}).items():
        key = int(k)
        if key not in shared.opp or v[2] > shared.opp[key][2]:
            shared.opp[key] = tuple(v)
    for k, v in p.get("mate", {}).items():
        key = int(k)
        if key not in shared.mate or v[2] > shared.mate[key][2]:
            shared.mate[key] = tuple(v)


def _prune(shared: Belief, now: int) -> None:
    if shared.ball and now - shared.ball[2] > STALE:
        shared.ball = None
    shared.opp = {k: v for k, v in shared.opp.items() if now - v[2] <= STALE}
    shared.mate = {k: v for k, v in shared.mate.items() if now - v[2] <= STALE}


class ArtelGossip:
    """Holds a team's running shared belief, refreshed each cycle by posting line sightings to Artel
    and polling them back. One instance per Artel-coached team per match."""

    def __init__(self, artel, team: str) -> None:
        self.artel = artel
        self.team = team
        self.since = EPOCH
        self.shared = Belief()

    async def cycle(self, line_beliefs: dict[str, Belief], now: int) -> Belief:
        for line, b in line_beliefs.items():
            await self.artel.emit_event(
                line, EVENT_TYPE, {"team": self.team, "line": line, **_ser(b)}
            )
        rows = await self.artel.poll_events("captain", EVENT_TYPE, self.since)
        for r in rows:
            self.since = max(self.since, r.get("created_at", self.since))
            p = r.get("payload", {})
            if p.get("team") != self.team:  # only fuse our own team's sightings
                continue
            _merge_into(self.shared, p)
        _prune(self.shared, now)
        return self.shared
