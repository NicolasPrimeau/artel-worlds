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
        self._resolve_possession()
        intents = {p.id: brain(self, p) for p in self.players}
        for p in self.players:
            self._move(p, intents[p.id])
        self._advance_ball(intents)
        self.tick += 1

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
        ux, uy = _unit(dx, dy)
        # accelerate toward the target heading rather than snapping — keeps motion smooth/legible
        p.vx += (ux * cap - p.vx) * c.accel
        p.vy += (uy * cap - p.vy) * c.accel
        sp = _len(p.vx, p.vy)
        if sp > cap:
            p.vx, p.vy = p.vx / sp * cap, p.vy / sp * cap
        # don't overshoot a near target (a keeper settling on a spot)
        if _len(dx, dy) < sp:
            p.vx, p.vy = dx, dy
        p.x = _clamp(p.x + p.vx, 0.0, c.length)
        p.y = _clamp(p.y + p.vy, 0.0, c.width)

    def _advance_ball(self, intents: dict) -> None:
        c = self.cfg
        b = self.ball
        if self.possessor is not None:
            owner = self.players[self.possessor]
            kick = intents[owner.id].get("kick")
            b.last_touch = owner.team
            if kick is not None:
                b.vx, b.vy = kick
                b.x += b.vx
                b.y += b.vy
                self.possessor = None
            else:  # shielding/holding — ball stays at the owner's feet, slightly ahead
                ax = c.length if owner.team == "home" else 0.0
                fx, fy = _unit(ax - owner.x, c.width / 2 - owner.y)
                b.x, b.y = owner.x + fx * 1.2, owner.y + fy * 1.2
                b.vx = b.vy = 0.0
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
        # end lines: goal or goal-kick
        if b.x <= 0.0:
            if self._in_mouth(b.y):
                return self._goal("away")
            return self._restart_endline("home")
        if b.x >= c.length:
            if self._in_mouth(b.y):
                return self._goal("home")
            return self._restart_endline("away")
        # touchlines: bounce back in (a throw-in is overkill for the spike)
        if b.y <= 0.0:
            b.y, b.vy = 0.0, abs(b.vy) * 0.5
        elif b.y >= c.width:
            b.y, b.vy = c.width, -abs(b.vy) * 0.5

    def _goal(self, scorer: str) -> None:
        self.score[scorer] += 1
        self.events.append(
            f"t{self.tick}: GOAL {scorer} ({self.score['home']}-{self.score['away']})"
        )
        self._kickoff("away" if scorer == "home" else "home")

    def _restart_endline(self, defending: str) -> None:
        # goal kick for the team whose line it crossed — placed just out of the box, ball dead
        c = self.cfg
        x = 12.0 if defending == "home" else c.length - 12.0
        self.ball = Ball(x, c.width / 2, last_touch=("away" if defending == "home" else "home"))
        self.possessor = None
