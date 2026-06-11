from __future__ import annotations

from dataclasses import dataclass

# 6 axial hex directions (pointy-top), same convention as Automata.
AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def hex_distance(aq: int, ar: int, bq: int, br: int) -> int:
    dq, dr = aq - bq, ar - br
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


def dir_toward(q: int, r: int, tq: int, tr: int) -> int:
    """The hex direction (0-5) that most reduces distance toward (tq, tr)."""
    d0 = hex_distance(q, r, tq, tr)
    best, bi = -99, 0
    for i, (dq, dr) in enumerate(AXIAL_DIRS):
        prog = d0 - hex_distance(q + dq, r + dr, tq, tr)
        if prog > best:
            best, bi = prog, i
    return bi


def _cube_round(x: float, y: float, z: float) -> tuple[int, int]:
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return (int(rx), int(rz))


def hex_line(aq: int, ar: int, bq: int, br: int) -> list[tuple[int, int]]:
    """Supercover hex line a->b (inclusive), for line-of-sight: every cell the straight
    segment between the two centers touches. A plain rounded line can sidestep a wall the
    segment visibly grazes — which reads as shooting through the obstacle. Sampling the
    segment nudged to each side and taking the union closes those gaps."""
    n = hex_distance(aq, ar, bq, br)
    if n == 0:
        return [(aq, ar)]
    ax, az = aq, ar
    ay = -ax - az
    bx, bz = bq, br
    by = -bx - bz
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n + 1):
        t = i / n
        x, y, z = ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t
        for ex, ey, ez in ((1e-6, 2e-6, -3e-6), (-1e-6, -2e-6, 3e-6)):
            cell = _cube_round(x + ex, y + ey, z + ez)
            if cell not in seen:
                seen.add(cell)
                out.append(cell)
    return out


@dataclass
class Tank:
    id: int
    team: str
    q: int
    r: int
    heading: int = 0
    energy: float = 100.0
    cooldown: int = 0
    target: int = 0  # last enemy fired at (for the viz tracer/aim)
    controller: str = ""
