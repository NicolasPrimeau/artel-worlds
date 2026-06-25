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

INDOOR_THEMES = {"office", "dark"}

INDOOR_ROOM_NAMES: dict[str, list[str]] = {
    "office": [
        "Open Plan",
        "Conference Room",
        "Break Room",
        "Reception",
        "Print Bay",
        "Manager's Office",
        "Server Room",
        "Supply Closet",
    ],
    "dark": [
        "Hall",
        "Crypt",
        "Chamber",
        "Cellar",
        "Vault",
        "Sanctum",
        "Dungeon",
        "Antechamber",
    ],
}

INDOOR_ROOM_PROPS: dict[str, dict[str, list[str]]] = {
    "office": {
        "Open Plan": ["desk", "chair", "partition", "plant_pot"],
        "Conference Room": ["desk", "chair", "whiteboard"],
        "Break Room": ["chair", "coffee"],
        "Reception": ["desk", "chair", "plant_pot"],
        "Print Bay": ["cabinet", "copier"],
        "Manager's Office": ["desk", "chair", "cabinet"],
        "Server Room": ["cabinet"],
        "Supply Closet": ["cabinet"],
    },
    "dark": {
        "Hall": ["rock", "topiary"],
        "Crypt": ["rock"],
        "Chamber": ["rock", "bush"],
        "Cellar": ["cabinet", "rock"],
        "Vault": ["cabinet"],
        "Sanctum": ["topiary", "flower"],
        "Dungeon": ["rock"],
        "Antechamber": ["rock", "topiary"],
    },
}


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
    world_w: int,
    world_h: int,
    tiles: list[int],
    a: list[int],
    b: list[int],
    walkable: set[int] | None = None,
) -> list[list[int]]:
    if walkable is None:
        walkable = PATHLIKE
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
            if tiles[_idx(world_w, nx, ny)] not in walkable:
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
        "scatter": [("desk", 4), ("chair", 5), ("cabinet", 3), ("partition", 3), ("plant_pot", 2)],
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


def generate_indoor_world(
    rng: random.Random, theme: str = "office", step_count: int = 6
) -> WorldMap:
    th = THEMES.get(theme, THEMES["office"])
    room_names = INDOOR_ROOM_NAMES.get(theme, INDOOR_ROOM_NAMES["office"])
    room_props = INDOOR_ROOM_PROPS.get(theme, {})
    default_props = ["rock"] if theme == "dark" else ["desk", "chair"]

    w, h = 40, 60
    tiles = [HEDGE] * (w * h)

    _ROOM_MIN = 5
    s = 2 * _ROOM_MIN

    cells: list[tuple[int, int, int, int]] = [(1, 1, w - 2, h - 2)]
    n_rooms = min(len(room_names), step_count + 2)
    while len(cells) < n_rooms:
        cands = sorted(
            (i for i in range(len(cells)) if cells[i][2] >= s or cells[i][3] >= s),
            key=lambda i: -(cells[i][2] * cells[i][3]),
        )
        if not cands:
            break
        cx, cy, cw, ch = cells.pop(cands[rng.randrange(min(3, len(cands)))])
        if cw >= s and (ch < s or cw >= ch or rng.random() < 0.5):
            cut = rng.randint(_ROOM_MIN, cw - _ROOM_MIN)
            cells += [(cx, cy, cut, ch), (cx + cut, cy, cw - cut, ch)]
        else:
            cut = rng.randint(_ROOM_MIN, ch - _ROOM_MIN)
            cells += [(cx, cy, cw, cut), (cx, cy + cut, cw, ch - cut)]

    rooms: list[tuple[int, int, int, int]] = []
    for cx, cy, cw, ch in cells:
        rw = min(cw - 2, max(_ROOM_MIN - 2, cw - rng.randint(1, max(1, cw // 3))))
        rh = min(ch - 2, max(_ROOM_MIN - 2, ch - rng.randint(1, max(1, ch // 3))))
        ox = rng.randint(1, cw - rw - 1) if cw - rw - 1 >= 1 else 1
        oy = rng.randint(1, ch - rh - 1) if ch - rh - 1 >= 1 else 1
        rooms.append((cx + ox, cy + oy, rw, rh))

    order = sorted(range(len(rooms)), key=lambda i: -(rooms[i][2] * rooms[i][3]))
    names = room_names[: len(rooms)]
    rects: dict[str, tuple[int, int, int, int]] = {
        names[k]: rooms[order[k]] for k in range(len(rooms))
    }

    room_floor = FLOOR if theme == "office" else th["base"]
    room_tiles: set[tuple[int, int]] = set()
    for name, (rx, ry, rw, rh) in rects.items():
        for ty in range(ry, ry + rh):
            for tx in range(rx, rx + rw):
                tiles[_idx(w, tx, ty)] = room_floor
                room_tiles.add((tx, ty))

    centers: dict[str, tuple[int, int]] = {
        n: (rects[n][0] + rects[n][2] // 2, rects[n][1] + rects[n][3] // 2) for n in names
    }

    corr: set[tuple[int, int]] = set()

    def carve_hallway(a: str, b: str) -> None:
        ax, ay = centers[a]
        bx, by = centers[b]
        if rng.random() < 0.5:
            for tx in range(min(ax, bx), max(ax, bx) + 1):
                if (tx, ay) not in room_tiles:
                    corr.add((tx, ay))
            for ty in range(min(ay, by), max(ay, by) + 1):
                if (bx, ty) not in room_tiles:
                    corr.add((bx, ty))
        else:
            for ty in range(min(ay, by), max(ay, by) + 1):
                if (ax, ty) not in room_tiles:
                    corr.add((ax, ty))
            for tx in range(min(ax, bx), max(ax, bx) + 1):
                if (tx, by) not in room_tiles:
                    corr.add((tx, by))

    intree: set[str] = {names[0]}
    while len(intree) < len(names):
        best = min(
            (
                (centers[a][0] - centers[b][0]) ** 2 + (centers[a][1] - centers[b][1]) ** 2,
                a,
                b,
            )
            for a in intree
            for b in names
            if b not in intree
        )
        carve_hallway(best[1], best[2])
        intree.add(best[2])

    for tx, ty in corr:
        if 0 < tx < w - 1 and 0 < ty < h - 1:
            tiles[_idx(w, tx, ty)] = PATH

    props: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    waypoints_by_y = sorted(names, key=lambda n: -centers[n][1])
    waypoints: list[list[int]] = [[centers[n][0], centers[n][1]] for n in waypoints_by_y]
    waypoints = waypoints[: step_count + 1]
    used_names = waypoints_by_y[: step_count + 1]

    for name in used_names:
        cx, cy = centers[name]
        tiles[_idx(w, cx, cy)] = LAVENDER
        props.append({"x": cx, "y": cy - 1, "kind": "lamp"})
        occupied.add((cx, cy))
        occupied.add((cx, cy - 1))

        kinds = room_props.get(name, default_props)
        rx, ry, rw, rh = rects[name]
        candidates = [
            (tx, ty)
            for tx in range(rx, rx + rw)
            for ty in range(ry, ry + rh)
            if (tx, ty) not in occupied and tiles[_idx(w, tx, ty)] == room_floor
        ]
        rng.shuffle(candidates)
        for placed, (px, py) in enumerate(candidates[:4]):
            props.append({"x": px, "y": py, "kind": kinds[placed % len(kinds)]})
            occupied.add((px, py))

    indoor_walkable = {room_floor, PATH, PATH_LIGHT, LAVENDER}
    route: list[list[int]] = []
    wp_route_idx: list[int] = []
    for i, wp in enumerate(waypoints):
        if i == 0:
            wp_route_idx.append(0)
            route.append(list(wp))
            continue
        seg = _bfs(w, h, tiles, waypoints[i - 1], wp, walkable=indoor_walkable)
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


def generate_world(rng: random.Random, theme: str = "garden", step_count: int = 6) -> WorldMap:
    if theme in INDOOR_THEMES:
        return generate_indoor_world(rng, theme=theme, step_count=step_count)
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
