from __future__ import annotations

from dataclasses import dataclass

# 8 compass directions, clockwise from North.
DIRS = ((0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1))
DIR_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def turn_toward(cur: int, target: int) -> int:
    """Rotate one step (of 8) from cur toward target; returns the new heading."""
    if cur == target:
        return cur
    diff = (target - cur) % 8
    return (cur + (1 if diff <= 4 else -1)) % 8


def bearing(dx: int, dy: int) -> int:
    """Nearest of the 8 directions pointing along (dx, dy)."""
    import math

    ang = math.atan2(dx, -dy)  # 0 = North, clockwise
    return round(ang / (math.pi / 4)) % 8


@dataclass
class Tank:
    id: int
    team: str
    x: int
    y: int
    heading: int = 0
    gun: int = 0
    energy: float = 100.0
    cooldown: int = 0
    controller: str = ""  # who owns this tank (player agent_id or "house:<name>")


@dataclass
class Shell:
    x: int
    y: int
    dx: int
    dy: int
    power: float
    team: str
    shooter: int
