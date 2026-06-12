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


def _bfs_first_step(
    q: int,
    r: int,
    tq: int,
    tr: int,
    blocked: set,
    R: int,
    avoid_first: frozenset | set,
) -> int | None:
    from collections import deque

    start, target = (q, r), (tq, tr)
    if start == target:
        return None

    def on_map(c: tuple) -> bool:
        return hex_distance(c[0], c[1], R, R) <= R

    target_enterable = on_map(target) and target not in blocked
    prev: dict = {start: None}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for dq_, dr_ in AXIAL_DIRS:
            n = (cur[0] + dq_, cur[1] + dr_)
            if n in prev or not on_map(n):
                continue
            if n in blocked:
                continue
            if cur == start and n in avoid_first:
                continue
            prev[n] = cur
            arrived = n == target if target_enterable else hex_distance(n[0], n[1], tq, tr) <= 1
            if arrived:
                node = n
                while prev[node] != start:
                    node = prev[node]
                return AXIAL_DIRS.index((node[0] - q, node[1] - r))
            queue.append(n)
    return None


def bfs_step(
    q: int,
    r: int,
    tq: int,
    tr: int,
    blocked: set,
    R: int,
    avoid_first: frozenset | set = frozenset(),
    soft: frozenset | set = frozenset(),
) -> int | None:
    """First-step direction (0-5) of a shortest known path from (q,r) toward (tq,tr) on the
    radius-R hexagon, around `blocked` cells (known walls). `soft` cells (tanks) are routed
    AROUND when any alternative path exists; only when they sit in the sole corridor does the
    path go through them, and then they merely veto the immediate step (they move, so the
    route stays valid — queue behind, don't stall). If the target itself cannot be entered,
    any cell beside it counts as arrival. Returns None when no route is known (caller falls
    back to a greedy step)."""
    if soft:
        step = _bfs_first_step(q, r, tq, tr, blocked | set(soft), R, frozenset())
        if step is not None:
            return step
    return _bfs_first_step(q, r, tq, tr, blocked, R, set(avoid_first) | set(soft))


@dataclass
class Tank:
    id: int
    team: str
    q: int
    r: int
    heading: int = 0
    energy: float = 100.0
    cooldown: int = 0
    hit_taken: float = 0.0  # damage received in the last resolved step
    hit_from: int = 0  # who landed it (tank id), for 'taking fire' awareness
    last_fire: str = ""  # what the last trigger pull actually did — the learning signal
    target: int = 0  # last enemy fired at (for the viz tracer/aim)
    controller: str = ""
