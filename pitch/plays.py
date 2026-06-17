from __future__ import annotations

# Coordinated PLAYS with follow-through. A play is a short state machine spanning several ticks and
# two players: once it fires, those players COMMIT to their scripted parts (instead of the reactive
# baseline) until it completes or breaks. This is the execution layer the baseline lacks — the thing
# that makes a coordinated move actually get carried out, not just hinted at by a positional bias.
#
# First play: the give-and-go (one-two). The carrier plays it to a wall and SPRINTS past the
# defender; the wall first-times it back into the run. Other players keep their baseline shape, so
# only the two combine — legible, and it can't break the rest of the team.

from .bot import _open, _pass, decide
from .engine import Pitch, Player, _len, _unit

GG_TTL = 42  # ticks a give-and-go may run before it's abandoned


def _fwd(team: str) -> float:
    return 1.0 if team == "home" else -1.0


def _adv(team: str, x: float, length: float) -> float:
    return x if team == "home" else length - x


def try_give_and_go(pitch: Pitch, team: str):
    # fire when our carrier, in the attacking half, has a defender just ahead to beat and an open
    # team-mate to bounce it off, with space behind the defender to run into
    poss = pitch.possessor
    if poss is None:
        return None
    carrier = pitch.players[poss]
    if carrier.team != team or carrier.role == "GK":
        return None
    c = pitch.cfg
    fwd = _fwd(team)
    if _adv(team, carrier.x, c.length) < c.length * 0.45:  # only in the front ~55%
        return None
    opps = [o for o in pitch.opponents(carrier) if o.role != "GK"]
    ahead = [o for o in opps if 2 < (o.x - carrier.x) * fwd < 16 and abs(o.y - carrier.y) < 13]
    if not ahead:
        return None
    defender = min(ahead, key=lambda o: _len(o.x - carrier.x, o.y - carrier.y))
    mates = [q for q in pitch.teammates(carrier) if q.id != carrier.id and q.role != "GK"]
    walls = [
        q
        for q in mates
        if 7 < _len(q.x - carrier.x, q.y - carrier.y) < 22
        and (q.x - carrier.x) * fwd < 6  # level or slightly behind — a real wall
        and _open(pitch, q) > 5.5
    ]
    if not walls:
        return None
    wall = min(walls, key=lambda q: _len(q.x - carrier.x, q.y - carrier.y))
    # run into the space PAST the defender, but kept ONSIDE — short of the second-to-last opponent,
    # so the return through-ball isn't flagged offside
    opp_adv = sorted((_adv(team, o.x, c.length) for o in pitch.opponents(carrier)), reverse=True)
    offside = opp_adv[1] if len(opp_adv) >= 2 else c.length
    run_adv = min(_adv(team, defender.x, c.length) + 11, offside - 2.0)
    if run_adv <= _adv(team, carrier.x, c.length) + 3:  # no room to actually penetrate -> skip
        return None
    rx = run_adv if team == "home" else c.length - run_adv
    side = 1.0 if defender.y < c.width / 2 else -1.0
    ry = min(max(defender.y + side * 8, 8.0), c.width - 8.0)
    if any(_len(o.x - rx, o.y - ry) < 6 for o in opps):  # need space at the run target
        return None
    return GiveAndGo(carrier.id, wall.id, (rx, ry), pitch.tick)


class GiveAndGo:
    kind = "give-and-go"

    def __init__(self, carrier: int, wall: int, run: tuple, t0: int) -> None:
        self.carrier = carrier
        self.wall = wall
        self.run = run
        self.t0 = t0
        self.phase = "to_wall"  # to_wall -> return -> done
        self._passed = False

    def players(self) -> tuple:
        return (self.carrier, self.wall)

    def advance(self, pitch: Pitch) -> None:
        poss = pitch.possessor
        if poss is None:
            return
        if self.phase == "to_wall" and poss == self.wall:
            self.phase = "return"
        elif self.phase == "return" and poss == self.carrier:
            self.phase = "done"

    def dead(self, pitch: Pitch) -> bool:
        if self.phase == "done":
            return True
        if pitch.tick - self.t0 > GG_TTL:
            return True
        poss = pitch.possessor
        own = pitch.players[self.carrier].team
        return poss is not None and pitch.players[poss].team != own  # lost the ball -> abort

    def intent(self, pitch: Pitch, p: Player) -> dict | None:
        b = pitch.ball
        if self.phase == "to_wall":
            if p.id == self.carrier:
                if pitch.possessor == self.carrier and not self._passed:
                    self._passed = True
                    return _pass(pitch, p, pitch.players[self.wall])  # lay it off to the wall
                return {"move": self.run, "sprint": True}  # ...and burst into the space
            if p.id == self.wall:
                return {"move": (b.x, b.y), "sprint": True}  # come to receive the lay-off
        elif self.phase == "return":
            if p.id == self.carrier:
                return {"move": self.run, "sprint": True}  # keep bursting past the man
            if p.id == self.wall and pitch.possessor == self.wall:
                # return it to the carrier ONLY once they've actually got past the man (advanced into
                # the run); until then hold it a beat — that's what makes it a penetrating one-two
                carrier = pitch.players[self.carrier]
                if _len(carrier.x - self.run[0], carrier.y - self.run[1]) < 9:
                    return _pass(pitch, p, carrier)
                return {"move": (p.x, p.y)}  # hold, shielding
            if p.id == self.wall:
                return {"move": self.run, "sprint": False}
        return None


class PlayManager:
    def __init__(self, team: str) -> None:
        self.team = team
        self.play: GiveAndGo | None = None
        self.started = 0
        self.completed = 0
        self._cd = 0  # cooldown ticks before another play may fire (no stutter)
        self.history: list[tuple] = []  # (start_tick, end_tick, (carrier, wall), outcome)

    def update(self, pitch: Pitch) -> None:
        if self._cd > 0:
            self._cd -= 1
        if self.play is not None:
            self.play.advance(pitch)
            if self.play.phase == "done":
                self.completed += 1
                self.history.append((self.play.t0, pitch.tick, self.play.players(), "completed"))
            if self.play.dead(pitch):
                if self.play.phase != "done":
                    self.history.append((self.play.t0, pitch.tick, self.play.players(), "aborted"))
                self.play = None
                self._cd = 35
        if self.play is None and self._cd <= 0:
            new = try_give_and_go(pitch, self.team)
            if new is not None:
                self.play = new
                self.started += 1

    def intent(self, pitch: Pitch, p: Player) -> dict | None:
        if self.play is not None and p.id in self.play.players():
            return self.play.intent(pitch, p)
        return None


def play_brain(team: str, mgr: PlayManager | None = None):
    mgr = mgr or PlayManager(team)
    st = {"tick": -1}

    def brain(pitch: Pitch, p: Player) -> dict:
        if pitch.tick != st["tick"]:
            st["tick"] = pitch.tick
            mgr.update(pitch)
        if p.team == team:
            it = mgr.intent(pitch, p)
            if it is not None:
                return it
        return decide(pitch, p)

    return brain
