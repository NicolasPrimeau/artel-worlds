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
    vx: float = 0.0
    vy: float = 0.0


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
    _rng: Random = field(default_factory=lambda: Random(0))

    def __post_init__(self) -> None:
        self._rng = Random(f"pitch:{self.seed}")

    # --- setup ---
    def _formation(self, team: str) -> list[tuple[str, float, float]]:
        # a 2-1-1 in front of a keeper, mirrored by side. x in own half; y spread across width.
        c = self.cfg
        w = c.width
        if team == "home":  # defends x=0, attacks x=length
            return [
                ("GK", 6.0, w / 2),
                ("DEF", 28.0, w * 0.32),
                ("DEF", 28.0, w * 0.68),
                ("MID", 52.0, w / 2),
                ("FWD", 78.0, w / 2),
            ]
        return [  # away: mirror across the halfway line
            ("GK", c.length - 6.0, w / 2),
            ("DEF", c.length - 28.0, w * 0.32),
            ("DEF", c.length - 28.0, w * 0.68),
            ("MID", c.length - 52.0, w / 2),
            ("FWD", c.length - 78.0, w / 2),
        ]

    def setup(self, home_names: list[str], away_names: list[str]) -> None:
        self.players = []
        pid = 0
        for team, names in (("home", home_names), ("away", away_names)):
            for (role, hx, hy), name in zip(self._formation(team), names):
                self.players.append(Player(pid, team, name, role, hx, hy, hx, hy))
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
        b = self.ball
        nearest, nd = None, 1e9
        for p in self.players:
            d = _len(p.x - b.x, p.y - b.y)
            if d < nd:
                nearest, nd = p, d
        self.possessor = nearest.id if (nearest and nd <= self.cfg.control_radius) else None

    def _move(self, p: Player, intent: dict) -> None:
        c = self.cfg
        tx, ty = intent.get("move", (p.x, p.y))
        cap = c.keeper_speed if p.role == "GK" else c.player_speed
        dx, dy = tx - p.x, ty - p.y
        dist = _len(dx, dy)
        ux, uy = _unit(dx, dy)
        # ARRIVE for positioning (ease to a stop, no jitter); SPRINT for ball-chasing (full pace,
        # so defenders actually close down). With the gentle accel both read as a real runner —
        # accelerating, gliding, arcing into turns.
        desired = cap if intent.get("sprint") else cap * min(1.0, dist / c.arrive_radius)
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
                ax = c.length if owner.team == "home" else 0.0
                fx, fy = _unit(ax - owner.x, c.width / 2 - owner.y)
                tx, ty = owner.x + fx * c.carry_ahead, owner.y + fy * c.carry_ahead
                b.vx = (tx - b.x) * c.carry_ease
                b.vy = (ty - b.y) * c.carry_ease
                b.x += b.vx
                b.y += b.vy
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
