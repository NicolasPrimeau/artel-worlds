from __future__ import annotations

from .engine import Pitch, Player, _clamp, _len, _unit

# A distributed-systems slice of the team brain, for measurement only (NOT wired into the server).
#
# Each player is a NODE with local state: it senses only entities within R_SENSE, keeps a decaying
# BELIEF of where things are, and decides using ONLY that belief + what teammates told it. There is
# no global pitch handed to a decision — the full field exists nowhere. The bus (Artel, here an
# in-process stand-in) is the ONLY way a node learns about things outside its own senses.
#
# Two teams run the SAME node brain; the only difference is whether the bus is on. Bus-on: nodes
# gossip sightings, so the team coordinates off the UNION of what its members see (press election,
# mark assignment and line height computed from shared belief — a replicated computation, no
# master). Bus-off: each node is blind beyond its own radius and acts alone. Cut the bus and the
# coordinated team collapses into the blind one — that's the wedge, by construction.

R_SENSE = 40.0  # a node perceives entities within this radius (pitch is 120x80) — tuned: the bus
# wedge is large and robust for R in ~30-120; it only inverts under degenerate fog (R<=18, where
# coordinating on near-empty belief is worse than reacting locally). 40 keeps both teams credible.
STALE = 45  # a belief older than this (ticks) is no longer trusted — the node has lost track
GOSSIP_EVERY = 12  # ticks between team-belief syncs (~1s, the live Artel cadence off the hot path).
# The wedge is robust to ~2s of latency and only inverts past ~4s; a node always overlays its OWN
# fresh senses on top of the shared cache, so only OFF-sensor info ages.
ROLE_ZONE = {"DEF": 0, "MID": 1, "FWD": 2}
LINE_OF = {"GK": "def", "DEF": "def", "MID": "mid", "FWD": "fwd"}  # which line agent gossips a node


def _fwd(team: str) -> float:
    return 1.0 if team == "home" else -1.0


def _adv(team: str, x: float, length: float) -> float:
    # distance up the attacking direction: 0 = own goal line. A threatening attacker has LOW adv.
    return x if team == "home" else length - x


def _pcost(team: str, role: str, px: float, py: float, ball: tuple, length: float) -> float:
    # press cost: distance to the ball, plus a penalty for leaving your zone (so a striker doesn't
    # track into his own box and a defender doesn't charge upfield) — the load-balancing weight.
    fx = ball[0] if team == "home" else length - ball[0]
    bz = 0 if fx < length / 3 else (1 if fx < 2 * length / 3 else 2)
    return _len(px - ball[0], py - ball[1]) + abs(bz - ROLE_ZONE.get(role, 1)) * 20.0


class Belief:
    __slots__ = ("ball", "opp", "mate")

    def __init__(self) -> None:
        self.ball: tuple | None = None  # (x, y, tick, team_in_possession_or_None)
        self.opp: dict[int, tuple] = {}  # id -> (x, y, tick)
        self.mate: dict[int, tuple] = {}  # id -> (x, y, tick)


class DistTeam:
    def __init__(self, team: str, bus_on: bool, artel_backed: bool = False) -> None:
        self.team = team
        self.bus_on = bus_on
        # artel_backed: the shared belief is filled ONLY by the async gossip loop reading Artel (live).
        # Otherwise it's an in-process merge of the nodes' beliefs (the measurement stand-in).
        self.artel_backed = artel_backed
        self.bel: dict[int, Belief] = {}  # per-node belief, persisted across ticks
        self._shared: Belief | None = None  # last gossiped team belief (refreshed on cadence)
        self._shared_at = -(10**9)

    # --- the tick: sense -> gossip -> coordinate -> decide ---
    def plan(self, pitch: Pitch, cache: dict) -> None:
        ids = [p.id for p in pitch.players if p.team == self.team]
        now = pitch.tick
        for pid in ids:
            self._sense(pitch, pid, now)
        if self.bus_on:
            # gossip on a cadence (real Artel has latency): refresh the shared team belief every
            # GOSSIP_EVERY ticks; between syncs the shared view ages, but each node overlays its OWN
            # current senses, so only information from OTHER nodes goes stale. When artel_backed, the
            # async gossip_cycle owns self._shared (read from Artel) — plan never fills it locally.
            if not self.artel_backed and now - self._shared_at >= GOSSIP_EVERY:
                self._shared = self._combine([self.bel[pid] for pid in ids])
                self._shared_at = now
            shared = self._shared or Belief()
            views = {pid: self._combine([shared, self.bel[pid]]) for pid in ids}
            assign, presser = self._coordinate(pitch, ids, shared)
            plans = {pid: (assign, presser) for pid in ids}
        else:
            views = {pid: self.bel[pid] for pid in ids}
            plans = {pid: self._coordinate_local(pitch, pid, self.bel[pid]) for pid in ids}
        for pid in ids:
            cache[pid] = self._decide(pitch, pid, views[pid], plans[pid])

    async def gossip_cycle(self, pitch: Pitch, gossip) -> None:
        # off the hot path (~1s, live only): aggregate each line's current sightings and hand them to
        # the Artel gossip, which posts them and polls the team's merged belief back. Sets self._shared
        # to whatever Artel returns — so if Artel is down, it stays empty and the team goes blind.
        now = pitch.tick
        lines: dict[str, Belief] = {"def": Belief(), "mid": Belief(), "fwd": Belief()}
        for pid, bel in self.bel.items():
            ln = LINE_OF.get(pitch.players[pid].role, "mid")
            lines[ln] = self._combine([lines[ln], bel])
        self._shared = await gossip.cycle(lines, now)

    def _sense(self, pitch: Pitch, pid: int, now: int) -> None:
        bel = self.bel.setdefault(pid, Belief())
        me = pitch.players[pid]
        for e in pitch.players:
            if e.id == pid or _len(me.x - e.x, me.y - e.y) <= R_SENSE:
                if e.team == self.team:
                    bel.mate[e.id] = (e.x, e.y, now)
                else:
                    bel.opp[e.id] = (e.x, e.y, now)
        b = pitch.ball
        if pitch.possessor == pid or _len(me.x - b.x, me.y - b.y) <= R_SENSE:
            team = pitch.players[pitch.possessor].team if pitch.possessor is not None else None
            bel.ball = (b.x, b.y, now, team)

    @staticmethod
    def _combine(beliefs: list[Belief]) -> Belief:
        # fuse beliefs, keeping the freshest sighting of each entity. Used both to gossip (combine all
        # nodes) and to overlay a node's own senses on the shared cache. A replicated view — every
        # node computes the same thing from the same posts, so coordination needs no master.
        v = Belief()
        bt = -1
        for b in beliefs:
            if b.ball and b.ball[2] > bt:
                bt, v.ball = b.ball[2], b.ball
            for oid, e in b.opp.items():
                if oid not in v.opp or e[2] > v.opp[oid][2]:
                    v.opp[oid] = e
            for mid, e in b.mate.items():
                if mid not in v.mate or e[2] > v.mate[mid][2]:
                    v.mate[mid] = e
        return v

    def _coordinate(self, pitch: Pitch, ids: list[int], view: Belief) -> tuple[dict, int | None]:
        # from the SHARED belief: elect one presser for the ball, and assign each remaining defender
        # a distinct attacker to mark (greedy nearest, goal-side) — no double-marks, no dropped man.
        now, L = pitch.tick, pitch.cfg.length
        ball = view.ball if (view.ball and now - view.ball[2] <= STALE) else None
        outfield = [pid for pid in ids if pitch.players[pid].role != "GK"]
        presser = None
        if ball:
            presser = min(
                outfield,
                key=lambda pid: _pcost(
                    self.team,
                    pitch.players[pid].role,
                    *(pitch.players[pid].x, pitch.players[pid].y),
                    ball,
                    L,
                ),
            )
        attackers = [
            (oid, e[0], e[1])
            for oid, e in view.opp.items()
            if now - e[2] <= STALE and _adv(self.team, e[0], L) < L * 0.6
        ]
        assign: dict[int, int] = {}
        used: set[int] = set()
        for pid in sorted(outfield, key=lambda d: _adv(self.team, pitch.players[d].x, L)):
            if pid == presser:
                continue
            me = pitch.players[pid]
            cands = [a for a in attackers if a[0] not in used]
            if not cands:
                break
            a = min(cands, key=lambda a: _len(me.x - a[1], me.y - a[2]))
            assign[pid], _ = a[0], used.add(a[0])
        return assign, presser

    def _coordinate_local(self, pitch: Pitch, pid: int, bel: Belief) -> tuple[dict, int | None]:
        # bus-off: the node decides ALONE from its own senses. It presses if it sees the ball and
        # believes it's nearest among the teammates it can see (so several may all think so -> herd),
        # and marks the nearest attacker it personally sees (so two can grab one, and unseen men run free).
        now, L = pitch.tick, pitch.cfg.length
        me = pitch.players[pid]
        ball = bel.ball if (bel.ball and now - bel.ball[2] <= STALE) else None
        presser = None
        if ball and me.role != "GK":
            mine = _pcost(self.team, me.role, me.x, me.y, ball, L)
            seen_mates = [
                (mid, e) for mid, e in bel.mate.items() if mid != pid and now - e[2] <= STALE
            ]
            if all(
                mine <= _pcost(self.team, pitch.players[mid].role, e[0], e[1], ball, L)
                for mid, e in seen_mates
            ):
                presser = pid
        assign: dict[int, int] = {}
        if presser != pid and me.role != "GK":
            atts = [
                (oid, e[0], e[1])
                for oid, e in bel.opp.items()
                if now - e[2] <= STALE and _adv(self.team, e[0], L) < L * 0.6
            ]
            if atts:
                a = min(atts, key=lambda a: _len(me.x - a[1], me.y - a[2]))
                assign[pid] = a[0]
        return assign, presser

    def _decide(self, pitch: Pitch, pid: int, view: Belief, plan: tuple) -> dict:
        me = pitch.players[pid]
        c = pitch.cfg
        now = pitch.tick
        assign, presser = plan
        if (
            pitch.pass_to == pid
        ):  # a pass is coming to me — run onto it (the ball is on my doorstep)
            return {"move": (pitch.ball.x, pitch.ball.y), "sprint": True}
        if me.role == "GK":
            return self._gk(pitch, me, view)
        if pitch.possessor == pid:
            return self._on_ball(pitch, me, view)
        ball = view.ball if (view.ball and now - view.ball[2] <= STALE) else None
        if ball and ball[3] == self.team:  # we believe we have it — get into a supporting position
            return {"move": self._support(pitch, me, view, ball), "kick": None}
        if presser == pid:
            if ball:
                return {"move": (ball[0], ball[1]), "sprint": True}
            return {"move": self._slot(pitch, me, ball)}
        mid = assign.get(pid)
        if mid is not None and mid in view.opp and now - view.opp[mid][2] <= STALE:
            ox, oy, _ = view.opp[mid]
            own_x = 0.0 if self.team == "home" else c.length
            ux, uy = _unit(own_x - ox, c.width / 2 - oy)  # sit a step goal-side of the man
            return {
                "move": (_clamp(ox + ux * 4, 6, c.length - 6), _clamp(oy + uy * 4, 6, c.width - 6))
            }
        return {"move": self._slot(pitch, me, ball)}

    def _slot(self, pitch: Pitch, me: Player, ball: tuple | None) -> tuple[float, float]:
        c = pitch.cfg
        L = c.length
        bx = ball[0] if ball else L / 2
        push = {"DEF": 0.35, "MID": 0.7, "FWD": 1.0}.get(me.role, 0.5)
        fx = me.home_x + (bx - L / 2) * push
        if me.role == "DEF":  # a back line stays goal-side of where it believes the ball is
            fx = min(fx, bx - 4) if self.team == "home" else max(fx, bx + 4)
        return (_clamp(fx, 8.0, L - 8.0), _clamp(me.home_y, 6.0, c.width - 6.0))

    def _support(self, pitch: Pitch, me: Player, view: Belief, ball: tuple) -> tuple[float, float]:
        # off-ball, our team has it: get into a position to RECEIVE. Mids and forwards push ahead of
        # the believed ball to offer a forward outlet (keeping their lane for width); defenders hold a
        # touch deeper for balance. De-stack off the nearest team-mate so we spread, not clump. This
        # is a LOCAL behaviour — it needs no bus — so even a blind team still builds and looks like
        # it's playing; the bus only makes the carrier AWARE of more of these options.
        c = pitch.cfg
        L = c.length
        now = pitch.tick
        bx = ball[0]
        fwd = _fwd(self.team)
        push = {"DEF": 0.45, "MID": 0.85, "FWD": 1.15}.get(me.role, 0.6)
        fx = me.home_x + (bx - L / 2) * push
        fy = me.home_y
        if me.role in ("MID", "FWD"):  # get ahead of the ball to be a forward pass option
            fx = fx * 0.7 + (bx + 20 * fwd) * 0.3
        if me.role == "DEF":
            fx = min(fx, bx - 4) if self.team == "home" else max(fx, bx + 4)
        mates = [
            (e[0], e[1]) for mid, e in view.mate.items() if mid != me.id and now - e[2] <= STALE
        ]
        near = min(mates, key=lambda mm: _len(fx - mm[0], fy - mm[1]), default=None)
        if near is not None and _len(fx - near[0], fy - near[1]) < 8.0:
            ax, ay = _unit(fx - near[0], fy - near[1])
            fx, fy = fx + ax * 5, fy + ay * 5
        return (_clamp(fx, 8.0, L - 8.0), _clamp(fy, 6.0, c.width - 6.0))

    def _on_ball(self, pitch: Pitch, me: Player, view: Belief) -> dict:
        c = pitch.cfg
        now = pitch.tick
        gx, gy = pitch.attack_goal(self.team)
        fwd = _fwd(self.team)
        dist_goal = _len(gx - me.x, gy - me.y)
        opps = [e for e in view.opp.values() if now - e[2] <= STALE]
        near = min((_len(me.x - e[0], me.y - e[1]) for e in opps), default=99.0)
        if dist_goal < c.shoot_range and (near > 3.0 or dist_goal < 13.0):
            ux, uy = _unit(gx - me.x, gy - me.y)
            return {"move": (gx, gy), "kick": (ux * c.shot_speed, uy * c.shot_speed)}

        def openness(x: float, y: float) -> float:
            return min((_len(x - e[0], y - e[1]) for e in opps), default=99.0)

        mates = [
            (mid, e[0], e[1])
            for mid, e in view.mate.items()
            if mid != me.id and now - e[2] <= STALE
        ]
        pressured = near < 4.0

        def release(cands: list) -> dict:
            mid, x, y = max(
                cands, key=lambda m: (m[1] - me.x) * fwd * 0.5 + openness(m[1], m[2]) * 0.5
            )
            tx, ty = x + c.pass_lead * fwd, y
            ux, uy = _unit(tx - me.x, ty - me.y)
            return {
                "move": (tx, ty),
                "kick": (ux * c.pass_speed, uy * c.pass_speed),
                "pass_to": mid,
            }

        if pressured:
            safe = [(m, x, y) for (m, x, y) in mates if openness(x, y) > 5.0]
            if safe:
                return release(safe)
        else:
            fwd_opts = [
                (m, x, y) for (m, x, y) in mates if (x - me.x) * fwd > 7.0 and openness(x, y) > 6.0
            ]
            if fwd_opts and pitch._rng.random() < 0.3:
                return release(fwd_opts)
        return {
            "move": (gx, _clamp(me.y, 8.0, c.width - 8.0)),
            "sprint": near > 8.0 and dist_goal > c.shoot_range,
        }

    def _gk(self, pitch: Pitch, me: Player, view: Belief) -> dict:
        c = pitch.cfg
        now = pitch.tick
        own_x = 6.0 if me.team == "home" else c.length - 6.0
        if pitch.possessor == me.id:
            ux, uy = _unit((c.length * 0.5) - me.x, (c.width * 0.3) - me.y)
            return {
                "move": (c.length * 0.5, c.width * 0.3),
                "kick": (ux * c.shot_speed * 0.9, uy * c.shot_speed * 0.9),
            }
        ball = view.ball if (view.ball and now - view.ball[2] <= STALE) else None
        ty = ball[1] if ball else c.width / 2
        ty = _clamp(ty, c.width / 2 - c.goal_width / 2, c.width / 2 + c.goal_width / 2)
        return {
            "move": (own_x, ty),
            "sprint": bool(ball and _len(me.x - ball[0], me.y - ball[1]) < 25),
        }


def make_dist_brain(home: DistTeam, away: DistTeam):
    # one brain closure the engine can call per player; it runs each team's full distributed phase
    # once per tick (the first time it's asked that tick) and caches every node's intent.
    cache: dict[int, dict] = {}
    state = {"tick": -1}

    def brain(pitch: Pitch, p: Player) -> dict:
        if pitch.tick != state["tick"]:
            state["tick"] = pitch.tick
            cache.clear()
            home.plan(pitch, cache)
            away.plan(pitch, cache)
        return cache.get(p.id, {"move": (p.x, p.y)})

    return brain
