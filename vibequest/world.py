from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

# VibeQuest world — a walkable pastel promenade the AI party crosses as the quest advances.
# Wes Anderson framing: a centered vertical avenue, symmetric decor, one station per quest step.
# The party walks the avenue from the bottom; each completed step releases them to the next station.

# Ground tile ids (index into static/assets/ground_tiles.png):
GRASS = 0
PATH = 1
WATER = 2
HEDGE = 3  # blocking
FLOOR = 4
SAND = 5
LAVENDER = 6
PATH_LIGHT = 7

BLOCKING = {HEDGE}


@dataclass
class WorldMap:
    w: int
    h: int
    tiles: list[int]
    props: list[dict] = field(default_factory=list)
    waypoints: list[list[int]] = field(default_factory=list)
    route: list[list[int]] = field(default_factory=list)
    wp_route_idx: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "w": self.w,
            "h": self.h,
            "tiles": self.tiles,
            "props": self.props,
            "waypoints": self.waypoints,
            "route": self.route,
            "wp_route_idx": self.wp_route_idx,
        }


def _idx(w: int, x: int, y: int) -> int:
    return y * w + x


def _bfs(
    world_w: int, world_h: int, tiles: list[int], a: list[int], b: list[int]
) -> list[list[int]]:
    start, goal = (a[0], a[1]), (b[0], b[1])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        cx, cy = cur
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < world_w and 0 <= ny < world_h):
                continue
            if (nx, ny) in prev:
                continue
            if tiles[_idx(world_w, nx, ny)] in BLOCKING:
                continue
            prev[(nx, ny)] = cur
            q.append((nx, ny))
    if goal not in prev:
        return [list(start), list(goal)]
    path: list[list[int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        path.append([node[0], node[1]])
        node = prev[node]
    path.reverse()
    return path


def generate_world(rng: random.Random, step_count: int = 5) -> WorldMap:
    w, h = 25, 44
    cx = w // 2
    tiles = [GRASS] * (w * h)

    # border hedge ring
    for x in range(w):
        tiles[_idx(w, x, 0)] = HEDGE
        tiles[_idx(w, x, h - 1)] = HEDGE
    for y in range(h):
        tiles[_idx(w, 0, y)] = HEDGE
        tiles[_idx(w, w - 1, y)] = HEDGE

    # central avenue (3 wide) of pale path
    for y in range(1, h - 1):
        for dx in (-1, 0, 1):
            tiles[_idx(w, cx + dx, y)] = PATH

    props: list[dict] = []
    waypoints: list[list[int]] = []

    # one station per step plus the start, spaced up the avenue from the bottom
    n = step_count + 1
    top, bottom = 4, h - 5
    for i in range(n):
        t = i / (n - 1)
        y = round(bottom - t * (bottom - top))
        waypoints.append([cx, y])
        # 3x3 lavender plaza marking the station
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                tiles[_idx(w, cx + ddx, y + ddy)] = PATH_LIGHT
        tiles[_idx(w, cx, y)] = LAVENDER
        props.append({"x": cx, "y": y - 1, "kind": "lamp"})

    # symmetric flanking trees along the avenue + scattered bushes (mirrored)
    for y in range(3, h - 3, 4):
        if tiles[_idx(w, cx - 4, y)] == GRASS:
            props.append({"x": cx - 4, "y": y, "kind": "tree"})
        if tiles[_idx(w, cx + 3, y)] == GRASS:
            props.append({"x": cx + 3, "y": y, "kind": "tree"})
    for _ in range(14):
        side = rng.choice((-1, 1))
        bx = cx + side * rng.randint(3, w // 2 - 2)
        by = rng.randint(2, h - 3)
        if 0 < bx < w - 1 and tiles[_idx(w, bx, by)] == GRASS:
            props.append({"x": bx, "y": by, "kind": "bush"})

    # a small mirrored pond near the middle, framed off the avenue
    pond_y = h // 2
    for px in range(cx + 5, cx + 8):
        for py in range(pond_y - 1, pond_y + 2):
            if 0 < px < w - 1 and tiles[_idx(w, px, py)] == GRASS:
                tiles[_idx(w, px, py)] = WATER

    # full walking route, station to station
    route: list[list[int]] = []
    wp_route_idx: list[int] = []
    for i, wp in enumerate(waypoints):
        if i == 0:
            wp_route_idx.append(0)
            route.append(list(wp))
            continue
        seg = _bfs(w, h, tiles, waypoints[i - 1], wp)
        wp_route_idx.append(len(route) - 1 + len(seg) - 1)
        route.extend(seg[1:])

    return WorldMap(
        w=w,
        h=h,
        tiles=tiles,
        props=props,
        waypoints=waypoints,
        route=route,
        wp_route_idx=wp_route_idx,
    )


def facing_from_delta(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
