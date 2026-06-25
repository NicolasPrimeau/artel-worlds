from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

GRASS = 0
PATH = 1
WATER = 2
HEDGE = 3
FLOOR = 4
SAND = 5
LAVENDER = 6
PATH_LIGHT = 7

BLOCKING = {HEDGE, WATER}
PATHLIKE = {PATH, PATH_LIGHT, LAVENDER}
OUTDOOR_WALKABLE = {GRASS, SAND, PATH, PATH_LIGHT, LAVENDER, FLOOR}

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

# tile_id → frame index in tiles_interior.png (22-col grid, frame = row*22+col)
INDOOR_TILE_FRAMES: dict[str, dict[int, int]] = {
    "office": {
        FLOOR: 44,  # row 2 col 0: light warm peach floor
        PATH: 67,  # row 3 col 1: slightly different warm floor
        PATH_LIGHT: 23,  # row 1 col 1: lighter highlight floor
        LAVENDER: 110,  # row 5 col 0: warmest cream (waypoint marker)
        HEDGE: 275,  # row 12 col 11: very dark wall
    },
    "dark": {
        FLOOR: 187,  # row 8 col 11: mossy grey-green dungeon floor
        PATH: 188,  # row 8 col 12: slightly lighter grey-green
        PATH_LIGHT: 166,  # row 7 col 12: lighter grey-green
        LAVENDER: 224,  # row 10 col 4: warm orange (torch-lit waypoint)
        HEDGE: 319,  # row 14 col 11: very dark wall
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
    tileset: str = "ground"
    tile_frames: dict = field(default_factory=dict)

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
            "tileset": self.tileset,
            "tile_frames": self.tile_frames,
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


def generate_indoor_world(
    rng: random.Random, theme: str = "office", step_count: int = 6
) -> WorldMap:
    th = THEMES.get(theme, THEMES["office"])
    room_names = INDOOR_ROOM_NAMES.get(theme, INDOOR_ROOM_NAMES["office"])
    room_props = INDOOR_ROOM_PROPS.get(theme, {})
    default_props = ["rock"] if theme == "dark" else ["desk", "chair"]

    w, h = 56, 56
    tiles = [HEDGE] * (w * h)

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            tiles[_idx(w, x, y)] = FLOOR

    n = min(len(room_names), step_count + 1)
    used_names = room_names[:n]

    margin = 8
    band_h = (h - 2 * margin) / n
    waypoints: list[list[int]] = []
    for i in range(n):
        cy = int(h - margin - band_h * (i + 0.5))
        cx = (
            rng.randint(margin, w // 2 - 4)
            if i % 2 == 0
            else rng.randint(w // 2 + 4, w - margin - 2)
        )
        waypoints.append([cx, cy])

    # Scatter 2×2 HEDGE pillar clusters for visual structure
    pillar_spots: list[tuple[int, int]] = []
    for py in range(8, h - 8, 7):
        for px in range(8, w - 8, 7):
            pillar_spots.append((px, py))
    rng.shuffle(pillar_spots)
    for px, py in pillar_spots[:14]:
        jx = px + rng.randint(-2, 2)
        jy = py + rng.randint(-2, 2)
        if 3 <= jx < w - 4 and 3 <= jy < h - 4:
            tiles[_idx(w, jx, jy)] = HEDGE
            tiles[_idx(w, jx + 1, jy)] = HEDGE
            tiles[_idx(w, jx, jy + 1)] = HEDGE
            tiles[_idx(w, jx + 1, jy + 1)] = HEDGE

    # Clear pillars that landed on or near waypoints
    for wx, wy in waypoints:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = wx + dx, wy + dy
                if 2 <= nx < w - 2 and 2 <= ny < h - 2:
                    tiles[_idx(w, nx, ny)] = FLOOR

    # Mark waypoint tiles
    for wx, wy in waypoints:
        tiles[_idx(w, wx, wy)] = LAVENDER

    occupied: set[tuple[int, int]] = set()
    props: list[dict] = []

    for i, ([wx, wy], name) in enumerate(zip(waypoints, used_names)):
        props.append({"x": wx, "y": wy - 1, "kind": "lamp"})
        occupied.update([(wx, wy), (wx, wy - 1)])

        kinds = room_props.get(name, default_props)
        ring = [
            (wx - 2, wy - 1),
            (wx - 2, wy),
            (wx - 2, wy + 1),
            (wx + 2, wy - 1),
            (wx + 2, wy),
            (wx + 2, wy + 1),
            (wx - 1, wy - 2),
            (wx, wy - 2),
            (wx + 1, wy - 2),
            (wx - 1, wy + 2),
            (wx, wy + 2),
            (wx + 1, wy + 2),
        ]
        for j, (px, py) in enumerate(ring):
            if 2 <= px < w - 2 and 2 <= py < h - 2 and (px, py) not in occupied:
                if tiles[_idx(w, px, py)] == FLOOR:
                    props.append({"x": px, "y": py, "kind": kinds[j % len(kinds)]})
                    occupied.add((px, py))

    scatter_kinds = room_props.get(used_names[len(used_names) // 2], default_props)
    for _ in range(100):
        x, y = rng.randint(3, w - 4), rng.randint(3, h - 4)
        if (x, y) not in occupied and tiles[_idx(w, x, y)] == FLOOR:
            props.append({"x": x, "y": y, "kind": rng.choice(scatter_kinds)})
            occupied.add((x, y))

    indoor_walkable = {FLOOR, PATH, PATH_LIGHT, LAVENDER}
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

    frames = INDOOR_TILE_FRAMES.get(theme, INDOOR_TILE_FRAMES["office"])

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
        tileset="interior",
        tile_frames={str(k): v for k, v in frames.items()},
    )


def generate_outdoor_world(
    rng: random.Random, theme: str = "garden", step_count: int = 6
) -> WorldMap:
    th = THEMES.get(theme, THEMES["garden"])
    base = th["base"]
    w, h = 64, 60
    tiles = [base] * (w * h)

    for x in range(w):
        tiles[_idx(w, x, 0)] = HEDGE
        tiles[_idx(w, x, h - 1)] = HEDGE
    for y in range(h):
        tiles[_idx(w, 0, y)] = HEDGE
        tiles[_idx(w, w - 1, y)] = HEDGE

    n = step_count + 1
    margin = 7
    band_h = (h - 2 * margin) / n
    waypoints: list[list[int]] = []
    for i in range(n):
        cy = int(h - margin - band_h * (i + 0.5))
        cx = (
            rng.randint(margin, w // 2 - 4)
            if i % 2 == 0
            else rng.randint(w // 2 + 4, w - margin - 2)
        )
        waypoints.append([cx, cy])

    occupied: set[tuple[int, int]] = set()
    props: list[dict] = []

    for wx, wy in waypoints:
        tiles[_idx(w, wx, wy)] = LAVENDER
        props.append({"x": wx, "y": wy - 1, "kind": "lamp"})
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = wx + dx, wy + dy
                if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                    occupied.add((nx, ny))
                    if tiles[_idx(w, nx, ny)] == base:
                        tiles[_idx(w, nx, ny)] = PATH_LIGHT
        tiles[_idx(w, wx, wy)] = LAVENDER

    # Irregular ponds (blob growth)
    if th["ponds"]:
        for _ in range(8):
            px, py = rng.randint(5, w - 6), rng.randint(5, h - 6)
            if (px, py) in occupied:
                continue
            blob: list[tuple[int, int]] = [(px, py)]
            for _ in range(rng.randint(4, 14)):
                bx, by = rng.choice(blob)
                nx, ny = bx + rng.randint(-1, 1), by + rng.randint(-1, 1)
                if 2 <= nx < w - 2 and 2 <= ny < h - 2 and (nx, ny) not in occupied:
                    blob.append((nx, ny))
            for bx, by in blob:
                if (bx, by) not in occupied:
                    tiles[_idx(w, bx, by)] = WATER
                    occupied.add((bx, by))

    # Compute route first so props don't block it
    route: list[list[int]] = []
    wp_route_idx: list[int] = []
    route_tiles: set[tuple[int, int]] = set()
    for i, wp in enumerate(waypoints):
        if i == 0:
            wp_route_idx.append(0)
            route.append(list(wp))
            continue
        seg = _bfs(w, h, tiles, waypoints[i - 1], wp, walkable=OUTDOOR_WALKABLE)
        wp_route_idx.append(len(route) - 1 + len(seg) - 1)
        route.extend(seg[1:])
    for rx, ry in route:
        route_tiles.add((rx, ry))

    # Dense prop scatter — groves then broad fill, never on the route
    weighted: list[str] = [kind for kind, wt in th["scatter"] for _ in range(wt)]

    def can_place(x: int, y: int, fp2: bool = False) -> bool:
        if (x, y) in occupied or (x, y) in route_tiles:
            return False
        if not (1 <= x < w - 1 and 1 <= y < h - 1):
            return False
        if tiles[_idx(w, x, y)] != base:
            return False
        if fp2:
            return all(
                tiles[_idx(w, x + dx, y + dy)] == base
                and (x + dx, y + dy) not in occupied
                and (x + dx, y + dy) not in route_tiles
                for dx in range(2)
                for dy in range(2)
            )
        return True

    for _ in range(35):
        gx, gy = rng.randint(2, w - 3), rng.randint(2, h - 3)
        kind = rng.choice(weighted)
        fp2 = kind in ("tree", "tree2")
        for _ in range(rng.randint(3, 8)):
            nx, ny = gx + rng.randint(-4, 4), gy + rng.randint(-4, 4)
            if can_place(nx, ny, fp2):
                props.append({"x": nx, "y": ny, "kind": kind})
                occupied.add((nx, ny))

    for _ in range(500):
        x, y = rng.randint(1, w - 2), rng.randint(1, h - 2)
        kind = rng.choice(weighted)
        fp2 = kind in ("tree", "tree2")
        if can_place(x, y, fp2):
            props.append({"x": x, "y": y, "kind": kind})
            occupied.add((x, y))

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
    return generate_outdoor_world(rng, theme=theme, step_count=step_count)


def facing_from_delta(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
