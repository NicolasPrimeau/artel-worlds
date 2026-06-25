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
PATHLIKE = {PATH, PATH_LIGHT, LAVENDER}


@dataclass
class WorldMap:
    w: int
    h: int
    tiles: list[int]
    props: list[dict] = field(default_factory=list)
    waypoints: list[list[int]] = field(default_factory=list)
    route: list[list[int]] = field(default_factory=list)
    wp_route_idx: list[int] = field(default_factory=list)
    theme: str = "garden"
    tint: str = "#ffffff"

    def to_dict(self) -> dict:
        return {
            "w": self.w,
            "h": self.h,
            "tiles": self.tiles,
            "props": self.props,
            "waypoints": self.waypoints,
            "route": self.route,
            "wp_route_idx": self.wp_route_idx,
            "theme": self.theme,
            "tint": self.tint,
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
            if tiles[_idx(world_w, nx, ny)] not in PATHLIKE:
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


# Each theme sets the ground base, whether ponds appear, a colour grade the client
# multiplies over the world, and a weighted prop mix — so the map looks like the quest.
THEMES = {
    "woods": {
        "base": GRASS,
        "ponds": True,
        "tint": "#e7f1dd",
        "scatter": [
            ("tree", 5),
            ("tree2", 4),
            ("bush", 3),
            ("topiary", 1),
            ("flower", 2),
            ("rock", 1),
        ],
    },
    "garden": {
        "base": GRASS,
        "ponds": True,
        "tint": "#f7eee0",
        "scatter": [("topiary", 4), ("flower", 5), ("bush", 3), ("tree", 2), ("rock", 1)],
    },
    "office": {
        "base": FLOOR,
        "ponds": False,
        "tint": "#efe4cf",
        "scatter": [("rock", 2), ("bush", 1), ("topiary", 1)],
    },
    "concrete": {
        "base": SAND,
        "ponds": False,
        "tint": "#e2dfd5",
        "scatter": [("rock", 5), ("bush", 1)],
    },
    "dark": {
        "base": GRASS,
        "ponds": True,
        "tint": "#9098b8",
        "scatter": [("tree", 4), ("rock", 3), ("bush", 1), ("topiary", 1)],
    },
}


def pick_theme(hook: str, register: str = "") -> str:
    h = f"{hook} {register}".lower()

    def has(*ks: str) -> bool:
        return any(k in h for k in ks)

    if has(
        "haunt",
        "dragon",
        "lair",
        "dungeon",
        "ghost",
        "gothic",
        "horror",
        "crypt",
        "noir",
        "war",
        "espionage",
        "spy",
        "disaster",
        "crime",
    ):
        return "dark"
    if has("forest", "enchant", "woods", "plant", "nature", "fairy", "rooftop garden"):
        return "woods"
    if has("garage", "parking", "rooftop", "concrete", "basement", "level b", "loading dock"):
        return "concrete"
    if has(
        "account",
        "filing",
        "cabinet",
        "desk",
        "report",
        "tps",
        "office",
        "cubicle",
        "supply",
        "drive",
        "slack",
        "calendar",
        "spreadsheet",
        "break room",
        "microwave",
        "mug",
        "printer",
        "conference",
        "meeting",
        "scissors",
        "stapler",
    ):
        return "office"
    return "garden"


def _carve(tiles: list[int], w: int, h: int, x: int, y: int, tile: int = PATH) -> None:
    for dx in range(2):
        for dy in range(2):
            nx, ny = x + dx, y + dy
            if 0 < nx < w - 1 and 0 < ny < h - 1:
                tiles[_idx(w, nx, ny)] = tile


def generate_world(rng: random.Random, theme: str = "garden", step_count: int = 6) -> WorldMap:
    th = THEMES.get(theme, THEMES["garden"])
    base = th["base"]
    w, h = 40, 60
    tiles = [base] * (w * h)

    for x in range(w):
        tiles[_idx(w, x, 0)] = HEDGE
        tiles[_idx(w, x, h - 1)] = HEDGE
    for y in range(h):
        tiles[_idx(w, 0, y)] = HEDGE
        tiles[_idx(w, w - 1, y)] = HEDGE

    # winding waypoints from bottom to top, x meandering across the open space
    n = step_count + 1
    margin = 6
    waypoints: list[list[int]] = []
    cur_x = w // 2
    for i in range(n):
        t = i / (n - 1)
        y = round((h - margin) - t * (h - 2 * margin))
        if i == 0:
            x = w // 2
        else:
            cur_x += rng.choice((-1, 1)) * rng.randint(7, 13)
            cur_x = max(margin, min(w - margin - 2, cur_x))
            x = cur_x
        waypoints.append([x, y])

    # carve a winding 2-wide path (L-shaped legs) connecting the waypoints
    for i in range(1, n):
        ax, ay = waypoints[i - 1]
        bx, by = waypoints[i]
        for x in range(min(ax, bx), max(ax, bx) + 1):
            _carve(tiles, w, h, x, ay)
        for y in range(min(ay, by), max(ay, by) + 1):
            _carve(tiles, w, h, bx, y)

    props: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for wp in waypoints:
        x, y = wp
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                nx, ny = x + ddx, y + ddy
                if 0 < nx < w - 1 and 0 < ny < h - 1:
                    tiles[_idx(w, nx, ny)] = PATH_LIGHT
                    occupied.add((nx, ny))
        tiles[_idx(w, x, y)] = LAVENDER
        props.append({"x": x, "y": y - 1, "kind": "lamp"})
        occupied.add((x, y - 1))

    def is_base(x: int, y: int) -> bool:
        return 0 < x < w - 1 and 0 < y < h - 1 and tiles[_idx(w, x, y)] == base

    def block_base(x: int, y: int) -> bool:
        return all(is_base(x + dx, y + dy) for dx in range(2) for dy in range(2))

    # ponds, framed by sand
    if th["ponds"]:
        for _ in range(5):
            px, py = rng.randint(3, w - 5), rng.randint(3, h - 5)
            if not block_base(px, py) or not block_base(px + 1, py + 1):
                continue
            for dx in range(3):
                for dy in range(3):
                    if is_base(px + dx, py + dy):
                        tiles[_idx(w, px + dx, py + dy)] = WATER

    # dense, theme-weighted decoration scattered across the open ground (clustered)
    weighted: list[str] = []
    for kind, weight in th["scatter"]:
        weighted += [kind] * weight

    def place(x: int, y: int, kind: str, footprint: int = 1) -> None:
        if (x, y) in occupied:
            return
        ok = block_base(x, y) if footprint == 2 else is_base(x, y)
        if not ok:
            return
        props.append({"x": x, "y": y, "kind": kind})
        occupied.add((x, y))

    # groves: clusters of the heavier scatter kinds
    for _ in range(20):
        gx, gy = rng.randint(2, w - 3), rng.randint(2, h - 3)
        kind = rng.choice(weighted)
        fp = 2 if kind in ("tree", "tree2") else 1
        for _ in range(rng.randint(2, 5)):
            place(gx + rng.randint(-2, 2), gy + rng.randint(-2, 2), kind, fp)
    # broad scatter to fill the rest
    for _ in range(220):
        x, y = rng.randint(2, w - 2), rng.randint(2, h - 2)
        kind = rng.choice(weighted)
        place(x, y, kind, 2 if kind in ("tree", "tree2") else 1)

    # walking route along the carved path, station to station
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
        theme=theme,
        tint=th["tint"],
    )


def facing_from_delta(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
