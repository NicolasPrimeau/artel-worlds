from __future__ import annotations

import math
from dataclasses import dataclass, field
from random import Random

from .config import DEFAULT, Config


def _len(dx: float, dy: float) -> float:
    return math.hypot(dx, dy)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    d = _len(dx, dy)
    return (0.0, 0.0) if d < 1e-9 else (dx / d, dy / d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


DISP = {"GK": "GK", "DEF": "DF", "MID": "MF", "FWD": "FW"}  # short position labels for player names


@dataclass
class Ball:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    last_touch: str | None = None  # which team last played it (for restarts)
    last_kicker: int | None = None  # player id who last struck it (for the scorer's name)


@dataclass
class Player:
    id: int
    team: str  # "home" | "away"
    name: str
    role: str  # GK | DEF | MID | FWD
    x: float
    y: float
    home_x: float  # formation anchor — where this role lives when the ball is at midfield
    home_y: float
    number: int = 0  # shirt number, drawn on the dot so players are tellable apart
    vx: float = 0.0
    vy: float = 0.0
    # per-player attributes (multipliers around 1.0) — each one rolls a slightly different player,
    # so a quick striker, a wayward passer, or a sticky-handed keeper emerge match to match
    pace: float = 1.0  # top speed
    acc: float = 1.0  # passing accuracy (higher = tighter)
    finishing: float = 1.0  # shooting accuracy (a clinical striker vs a wayward one)
    control: float = 1.0  # first touch / reach to collect a loose ball
    strength: float = 1.0  # reach to win the ball in a duel / tackle an opponent
    handling: float = 1.0  # keeper save reach


@dataclass
class Pitch:
    """One match on a continuous 2D field. The engine is the dumb, deterministic motor: it
    resolves movement, possession, kicks, and goals. WHO chases, passes, or holds shape is the
    brain's job (bot.py for the baseline; an Artel commander on top later) — exactly the phalanx
    split. Home attacks toward +x; away toward -x."""

    cfg: Config = DEFAULT
    seed: int = 0
    players: list[Player] = field(default_factory=list)
    ball: Ball = field(default_factory=lambda: Ball(60.0, 40.0))
    score: dict[str, int] = field(default_factory=lambda: {"home": 0, "away": 0})
    tick: int = 0
    possessor: int | None = None  # player id currently on the ball, or None (loose)
    events: list[str] = field(default_factory=list)  # goals etc., for the feed
    celebrate: int = 0  # ticks remaining of a goal freeze (so the score reads before kickoff)
    restart: int = 0  # ticks remaining of a dead-ball pause (throw-in / corner / goal kick)
    scorer: str | None = None  # name shown during a celebration
    goal_team: str | None = None  # team that just scored (for the celebration colour)
    restart_kind: str | None = None  # "corner" | "goal-kick" | "throw-in" — for the feed
    _concede_to: str = "home"  # who kicks off after the celebration
    shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # team -> (def, mid, fwd)
    _rng: Random = field(default_factory=lambda: Random(0))

    def __post_init__(self) -> None:
        self._rng = Random(f"pitch:{self.seed}")

    # --- setup ---
    def _shape(self, out: int) -> tuple[int, int, int]:
        # a random but sensible (defenders, midfielders, forwards) split for the squad — so each
        # team lines up differently (e.g. 4-2-2 vs 3-2-3). >=2 at the back, >=1 in midfield/attack.
        cands = [
            (nd, out - nd - nf, nf)
            for nd in range(2, 5)
            for nf in range(1, 4)
            if 1 <= out - nd - nf <= 4
        ]
        if not cands:  # tiny squads — fall back to a balanced split
            nd = max(1, round(out * 0.38))
            nf = max(1, round(out * 0.34))
            return nd, max(0, out - nd - nf), nf
        return self._rng.choice(cands)

    def _positions(self, team: str, shape: tuple[int, int, int]) -> list[tuple[str, float, float]]:
        # a keeper plus three lines, spread across the width. Home defends x=0 and attacks +x.
        c = self.cfg
        w = c.length
        pos: list[tuple[str, float, float]] = [
            ("GK", 6.0 if team == "home" else c.length - 6.0, c.width / 2)
        ]
        for role, depth, cnt in zip(("DEF", "MID", "FWD"), (0.22, 0.44, 0.68), shape):
            if cnt <= 0:
                continue
            x = depth * w if team == "home" else w - depth * w
            for j in range(cnt):
                pos.append((role, x, c.width * (j + 1) / (cnt + 1)))
        return pos

    def setup(self, home_names: list[str], away_names: list[str]) -> None:
        # *_names are plain surnames (team_size each); the position label + shirt number come from
        # the formation slot, so squad size is driven entirely by cfg.team_size.
        self.players = []
        self.shapes = {}
        pid = 0
        r = self._rng
        out = self.cfg.team_size - 1
        for team, names in (("home", home_names), ("away", away_names)):
            shape = self._shape(out)
            self.shapes[team] = shape
            for slot, ((role, hx, hy), surname) in enumerate(
                zip(self._positions(team, shape), names)
            ):
                p = Player(pid, team, f"{DISP[role]} {surname}", role, hx, hy, hx, hy, slot + 1)
                p.pace = round(r.uniform(0.9, 1.12), 3)
                p.acc = round(r.uniform(0.85, 1.15), 3)
                p.finishing = round(r.uniform(0.85, 1.15), 3)
                p.control = round(r.uniform(0.92, 1.1), 3)
                p.strength = round(r.uniform(0.9, 1.12), 3)
                p.handling = round(r.uniform(0.86, 1.14), 3) if role == "GK" else 1.0
                self.players.append(p)
                pid += 1
        self._kickoff("home")

    def _kickoff(self, _conceding: str) -> None:
        self.ball = Ball(self.cfg.length / 2, self.cfg.width / 2, last_touch=None)
        self.possessor = None
        self.celebrate = self.restart = 0
        self.scorer = self.goal_team = self.restart_kind = None
        for p in self.players:
            p.x, p.y, p.vx, p.vy = p.home_x, p.home_y, 0.0, 0.0

    # --- queries used by the brain ---
    def teammates(self, p: Player) -> list[Player]:
        return [q for q in self.players if q.team == p.team]

    def opponents(self, p: Player) -> list[Player]:
        return [q for q in self.players if q.team != p.team]

    def attack_goal(self, team: str) -> tuple[float, float]:
        return (
            (self.cfg.length, self.cfg.width / 2) if team == "home" else (0.0, self.cfg.width / 2)
        )

    # --- the tick ---
    def step(self, brain) -> None:
        self.tick += 1
        if self.celebrate > 0:
            # GOAL freeze — everything holds so the score is readable, then kick off
            self.celebrate -= 1
            if self.celebrate == 0:
                self.scorer = self.goal_team = None
                self._kickoff(self._concede_to)
            return
        if self.restart > 0:
            # dead ball (corner/goal-kick/throw-in): players reposition, the ball sits
            self.restart -= 1
            intents = {p.id: brain(self, p) for p in self.players}
            for p in self.players:
                self._move(p, intents[p.id])
            if self.restart == 0:
                self.restart_kind = None
            return
        self._resolve_possession()
        intents = {p.id: brain(self, p) for p in self.players}
        for p in self.players:
            self._move(p, intents[p.id])
        self._advance_ball(intents)

    def _resolve_possession(self) -> None:
        # a player controls the ball when it's within their reach; the keeper's reach is larger
        # (a dive/parry), so on-target shots get gathered instead of trickling in. Closest
        # qualifying player wins, so an outfielder can still beat a keeper to a loose ball.
        b = self.ball
        c = self.cfg
        prev = self.players[self.possessor].team if self.possessor is not None else None
        best, bd = None, 1e9
        for p in self.players:
            d = _len(p.x - b.x, p.y - b.y)
            if p.role == "GK":
                reach = c.gk_reach * p.handling
            elif prev is not None and prev != p.team:
                reach = c.control_radius * p.strength  # tackling it off an opponent — a duel
            else:
                reach = c.control_radius * p.control  # collecting a loose or own ball
            if d <= reach and d < bd:
                best, bd = p, d
        self.possessor = best.id if best else None

    def _move(self, p: Player, intent: dict) -> None:
        c = self.cfg
        tx, ty = intent.get("move", (p.x, p.y))
        sprint = intent.get("sprint")
        cap = (c.keeper_speed if p.role == "GK" else c.player_speed) * p.pace
        if self.possessor == p.id:
            cap *= 0.9  # running WITH the ball is slightly slower — defenders can close down
        dx, dy = tx - p.x, ty - p.y
        dist = _len(dx, dy)
        ux, uy = _unit(dx, dy)
        # DRIBBLE around opponents — the player on the ball steers slightly around defenders in
        # their path (a jink, not a hard swerve) instead of running straight through them. Only the
        # carrier does this; defenders hold and block rather than politely stepping aside.
        if self.possessor == p.id:
            sx, sy = 0.0, 0.0
            for o in self.players:
                if o.team == p.team:
                    continue
                ox, oy = o.x - p.x, o.y - p.y
                od = _len(ox, oy)
                if 0.0 < od < 4.5 and (ux * ox + uy * oy) > 0:  # close AND ahead of us
                    px, py = -uy, ux  # perpendicular to our heading
                    side = -1.0 if (px * ox + py * oy) > 0 else 1.0  # step to the freer side
                    w = (4.5 - od) / 4.5
                    sx += px * side * w
                    sy += py * side * w
            if sx or sy:
                ux, uy = _unit(ux + sx * 0.6, uy + sy * 0.6)
        # MOMENTUM: changing direction costs speed — you can't turn at full pace, and the velocity
        # blends (it never snaps), so players carry through their runs and ease into turns.
        sp = _len(p.vx, p.vy)
        turn = 1.0
        if sp > 1e-6 and p.role != "GK":  # keepers stay sharp across their line
            align = (p.vx * ux + p.vy * uy) / sp  # 1 = same heading, -1 = reversing
            turn = 0.55 + 0.45 * max(0.0, align)
        desired = (cap if sprint else cap * min(1.0, dist / c.arrive_radius)) * turn
        p.vx += (ux * desired - p.vx) * c.accel
        p.vy += (uy * desired - p.vy) * c.accel
        sp = _len(p.vx, p.vy)
        if sp > cap:
            p.vx, p.vy = p.vx / sp * cap, p.vy / sp * cap
        p.x = _clamp(p.x + p.vx, 0.0, c.length)
        p.y = _clamp(p.y + p.vy, 0.0, c.width)

    def _advance_ball(self, intents: dict) -> None:
        c = self.cfg
        b = self.ball
        if self.possessor is not None:
            owner = self.players[self.possessor]
            kick = intents[owner.id].get("kick")
            b.last_touch = owner.team
            if kick is not None:  # pass or shot — struck away
                b.vx, b.vy = kick
                b.x += b.vx
                b.y += b.vy
                b.last_kicker = owner.id
                self.possessor = None
            else:
                # CARRY — the ball eases to a spot just ahead of the carrier instead of being
                # re-kicked every tick. Velocity is real (target-tracking), so it reads as a
                # smooth dribble and the client interpolates it cleanly. No more jitter.
                # lead the ball in the direction the carrier is actually running (so it follows
                # their jink and angle of attack), falling back to goalward when nearly stopped
                hx, hy = owner.vx, owner.vy
                if _len(hx, hy) < 0.05:
                    ax = c.length if owner.team == "home" else 0.0
                    hx, hy = ax - owner.x, c.width / 2 - owner.y
                fx, fy = _unit(hx, hy)
                tx, ty = owner.x + fx * c.carry_ahead, owner.y + fy * c.carry_ahead
                b.vx = (tx - b.x) * c.carry_ease
                b.vy = (ty - b.y) * c.carry_ease
                b.x += b.vx
                b.y += b.vy
                # you can't dribble it into the net — a carried ball is held just short of the goal
                # line, so a goal can only come from a struck ball (a shot). Soccer rules apply.
                b.x = min(b.x, c.length - 0.6) if owner.team == "home" else max(b.x, 0.6)
                b.last_kicker = owner.id
        else:
            b.x += b.vx
            b.y += b.vy
            b.vx *= c.ball_friction
            b.vy *= c.ball_friction
        self._boundaries()

    def _in_mouth(self, y: float) -> bool:
        m = self.cfg.goal_width / 2
        return abs(y - self.cfg.width / 2) <= m

    def _boundaries(self) -> None:
        c = self.cfg
        b = self.ball
        if b.x <= 0.0:
            return self._goal("away") if self._in_mouth(b.y) else self._restart_endline("home")
        if b.x >= c.length:
            return self._goal("home") if self._in_mouth(b.y) else self._restart_endline("away")
        if b.y <= 0.0 or b.y >= c.width:
            return self._throw_in()

    def _scorer_name(self) -> str | None:
        k = self.ball.last_kicker
        return self.players[k].name if k is not None and 0 <= k < len(self.players) else None

    def _goal(self, scorer: str) -> None:
        self.score[scorer] += 1
        who = self._scorer_name() or scorer
        self.events.append(
            f"t{self.tick}: GOAL — {who} ({self.score['home']}-{self.score['away']})"
        )
        # freeze and celebrate before kickoff — don't snap straight back to centre
        self.celebrate = self.cfg.celebrate_ticks
        self.scorer = who
        self.goal_team = scorer
        self._concede_to = "away" if scorer == "home" else "home"
        self.ball.vx = self.ball.vy = 0.0
        self.possessor = None

    def _dead_ball(self, x: float, y: float, to_team: str, kind: str) -> None:
        # place a dead ball and pause briefly so the restart reads, then play resumes
        self.ball = Ball(x, y, last_touch=("away" if to_team == "home" else "home"))
        self.possessor = None
        self.restart = self.cfg.restart_ticks
        self.restart_kind = kind

    def _restart_endline(self, defending: str) -> None:
        c = self.cfg
        b = self.ball
        out_top = b.y < c.width / 2
        if b.last_touch == defending:  # defender put it out → CORNER to the attackers
            x = 1.0 if defending == "home" else c.length - 1.0
            y = 1.0 if out_top else c.width - 1.0
            self._dead_ball(x, y, "away" if defending == "home" else "home", "corner")
        else:  # attacker put it out → GOAL KICK to the defenders
            x = 12.0 if defending == "home" else c.length - 12.0
            self._dead_ball(x, c.width / 2, defending, "goal-kick")

    def _throw_in(self) -> None:
        c = self.cfg
        b = self.ball
        y = 0.5 if b.y <= 0.0 else c.width - 0.5
        to_team = "away" if b.last_touch == "home" else "home"  # other team throws in
        self._dead_ball(_clamp(b.x, 6.0, c.length - 6.0), y, to_team, "throw-in")
