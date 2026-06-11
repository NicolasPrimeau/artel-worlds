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


def hex_line(aq: int, ar: int, bq: int, br: int) -> list[tuple[int, int]]:
    """Cells along the hex line a->b (inclusive), for line-of-sight."""
    n = hex_distance(aq, ar, bq, br)
    if n == 0:
        return [(aq, ar)]
    ax, az = aq, ar
    ay = -ax - az
    bx, bz = bq, br
    by = -bx - bz
    out = []
    for i in range(n + 1):
        t = i / n
        x, y, z = ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t
        rx, ry, rz = round(x), round(y), round(z)
        dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dy > dz:
            ry = -rx - rz
        else:
            rz = -rx - ry
        out.append((rx, rz))
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
