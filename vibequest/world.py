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

INDOOR_THEMES = {"office", "dark", "pub", "school", "grocery"}

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
    "pub": [
        "Common Room",
        "Private Booth",
        "Kitchen",
        "Cellar",
        "Innkeeper's Den",
        "Back Corridor",
        "Vault",
        "Storeroom",
    ],
    "school": [
        "Classroom",
        "Library",
        "Cafeteria",
        "Principal's Office",
        "Science Lab",
        "Gymnasium",
        "Storage Room",
        "Assembly Hall",
    ],
    "grocery": [
        "Produce Section",
        "Bakery",
        "Deli Counter",
        "Dairy Aisle",
        "Checkout",
        "Manager's Office",
        "Stock Room",
        "Loading Bay",
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
    waypoint_names: list[str] = field(default_factory=list)
    route: list[list[int]] = field(default_factory=list)
    wp_route_idx: list[int] = field(default_factory=list)
    theme: str = "garden"
    tint: str = "#ffffff"
    tileset: str = "ground"
    tile_frames: dict = field(default_factory=dict)
    walkable: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "w": self.w,
            "h": self.h,
            "tiles": self.tiles,
            "props": self.props,
            "waypoints": self.waypoints,
            "waypoint_names": self.waypoint_names,
            "route": self.route,
            "wp_route_idx": self.wp_route_idx,
            "theme": self.theme,
            "tint": self.tint,
            "tileset": self.tileset,
            "tile_frames": self.tile_frames,
            "walkable": self.walkable,
        }


def find_path(world: "WorldMap", ax: int, ay: int, bx: int, by: int) -> list[list[int]]:
    walk = set(world.walkable) if world.walkable else PATHLIKE
    return _bfs(world.w, world.h, world.tiles, [ax, ay], [bx, by], walkable=walk)


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
        "tint": "#c8e8b8",
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
        "tint": "#f8d8c8",
        "scatter": [("topiary", 4), ("flower", 5), ("bush", 3), ("tree", 2), ("rock", 1)],
    },
    "office": {
        "base": FLOOR,
        "ponds": False,
        "tint": "#ffd8c8",
        "scatter": [("desk", 4), ("chair", 5), ("cabinet", 3), ("partition", 3), ("plant_pot", 2)],
    },
    "concrete": {
        "base": SAND,
        "ponds": False,
        "tint": "#e8d890",
        "scatter": [("rock", 5), ("bush", 1)],
    },
    "dark": {
        "base": GRASS,
        "ponds": True,
        "tint": "#b0a8d8",
        "scatter": [("tree", 4), ("rock", 3), ("bush", 1), ("topiary", 1)],
    },
    "pub": {
        "base": FLOOR,
        "ponds": False,
        "tint": "#ffe0a0",
        "scatter": [("chair", 4), ("coffee", 3), ("plant_pot", 2), ("cabinet", 2)],
    },
    "school": {
        "base": FLOOR,
        "ponds": False,
        "tint": "#b8e0d0",
        "scatter": [("desk", 5), ("chair", 4), ("cabinet", 2), ("plant_pot", 2)],
    },
    "grocery": {
        "base": FLOOR,
        "ponds": False,
        "tint": "#c8e8b0",
        "scatter": [("cabinet", 5), ("plant_pot", 3), ("chair", 2)],
    },
}


_ZONE_DEF: dict[str, dict[str, tuple[int, int, int]]] = {
    "office": {
        "Open Plan": (FLOOR, 22, 12),
        "Conference Room": (LAVENDER, 22, 12),
        "Break Room": (PATH_LIGHT, 18, 10),
        "Reception": (PATH, 18, 10),
        "Print Bay": (PATH_LIGHT, 18, 10),
        "Manager's Office": (FLOOR, 18, 10),
        "Server Room": (PATH, 16, 8),
        "Supply Closet": (PATH_LIGHT, 16, 8),
    },
    "dark": {
        "Hall": (FLOOR, 22, 12),
        "Crypt": (PATH, 18, 10),
        "Chamber": (LAVENDER, 22, 12),
        "Cellar": (PATH, 18, 10),
        "Vault": (FLOOR, 16, 8),
        "Sanctum": (LAVENDER, 22, 12),
        "Dungeon": (PATH, 18, 10),
        "Antechamber": (PATH_LIGHT, 18, 10),
    },
    "pub": {
        "Common Room": (FLOOR, 22, 12),
        "Private Booth": (LAVENDER, 16, 10),
        "Kitchen": (PATH_LIGHT, 18, 10),
        "Cellar": (PATH, 18, 10),
        "Innkeeper's Den": (FLOOR, 16, 10),
        "Back Corridor": (PATH, 14, 8),
        "Vault": (PATH_LIGHT, 14, 8),
        "Storeroom": (PATH, 16, 8),
    },
    "school": {
        "Classroom": (FLOOR, 22, 12),
        "Library": (LAVENDER, 22, 12),
        "Cafeteria": (PATH_LIGHT, 20, 12),
        "Principal's Office": (FLOOR, 16, 10),
        "Science Lab": (PATH, 20, 10),
        "Gymnasium": (PATH_LIGHT, 22, 12),
        "Storage Room": (PATH, 14, 8),
        "Assembly Hall": (LAVENDER, 22, 12),
    },
    "grocery": {
        "Produce Section": (FLOOR, 22, 12),
        "Bakery": (PATH_LIGHT, 18, 10),
        "Deli Counter": (LAVENDER, 18, 10),
        "Dairy Aisle": (PATH, 18, 12),
        "Checkout": (FLOOR, 20, 10),
        "Manager's Office": (FLOOR, 16, 10),
        "Stock Room": (PATH, 16, 8),
        "Loading Bay": (PATH_LIGHT, 18, 10),
    },
}

_ZONE_ARCHETYPES: dict[str, str] = {
    "Open Plan": "work_area",
    "Conference Room": "meeting_space",
    "Break Room": "lounge",
    "Reception": "entrance",
    "Print Bay": "equipment_bay",
    "Manager's Office": "private_office",
    "Server Room": "storage",
    "Supply Closet": "storage",
    "Hall": "thoroughfare",
    "Crypt": "storage",
    "Chamber": "shrine",
    "Cellar": "storage",
    "Vault": "storage",
    "Sanctum": "shrine",
    "Dungeon": "thoroughfare",
    "Antechamber": "entrance",
    "Common Room": "lounge",
    "Private Booth": "meeting_space",
    "Kitchen": "equipment_bay",
    "Innkeeper's Den": "private_office",
    "Back Corridor": "thoroughfare",
    "Storeroom": "storage",
    "Classroom": "work_area",
    "Library": "meeting_space",
    "Cafeteria": "lounge",
    "Principal's Office": "private_office",
    "Science Lab": "equipment_bay",
    "Gymnasium": "thoroughfare",
    "Storage Room": "storage",
    "Assembly Hall": "shrine",
    "Produce Section": "lounge",
    "Bakery": "equipment_bay",
    "Deli Counter": "entrance",
    "Dairy Aisle": "thoroughfare",
    "Checkout": "work_area",
    "Stock Room": "storage",
    "Loading Bay": "thoroughfare",
}

_ARCHETYPE_PROPS: dict[str, dict[str, list[str]]] = {
    "work_area": {
        "office": ["desk", "chair", "partition"],
        "dark": ["rock", "rock", "topiary"],
        "pub": ["desk", "chair", "chair"],
        "school": ["desk", "chair", "whiteboard"],
        "grocery": ["desk", "chair", "partition"],
    },
    "meeting_space": {
        "office": ["desk", "chair", "whiteboard"],
        "dark": ["topiary", "rock", "flower"],
        "pub": ["chair", "chair", "desk"],
        "school": ["desk", "chair", "whiteboard"],
        "grocery": ["desk", "chair", "whiteboard"],
    },
    "lounge": {
        "office": ["chair", "coffee", "plant_pot"],
        "dark": ["topiary", "flower", "rock"],
        "pub": ["chair", "coffee", "plant_pot"],
        "school": ["chair", "plant_pot", "plant_pot"],
        "grocery": ["plant_pot", "chair", "plant_pot"],
    },
    "entrance": {
        "office": ["desk", "chair", "plant_pot"],
        "dark": ["rock", "lamp", "topiary"],
        "pub": ["desk", "chair", "plant_pot"],
        "school": ["desk", "chair", "plant_pot"],
        "grocery": ["desk", "chair", "plant_pot"],
    },
    "equipment_bay": {
        "office": ["copier", "cabinet", "plant_pot"],
        "dark": ["rock", "rock", "rock"],
        "pub": ["coffee", "cabinet", "plant_pot"],
        "school": ["cabinet", "desk", "plant_pot"],
        "grocery": ["cabinet", "cabinet", "plant_pot"],
    },
    "private_office": {
        "office": ["desk", "chair", "cabinet"],
        "dark": ["rock", "cabinet", "topiary"],
        "pub": ["desk", "chair", "plant_pot"],
        "school": ["desk", "chair", "cabinet"],
        "grocery": ["desk", "chair", "cabinet"],
    },
    "storage": {
        "office": ["cabinet", "cabinet", "cabinet"],
        "dark": ["rock", "cabinet", "rock"],
        "pub": ["cabinet", "plant_pot", "cabinet"],
        "school": ["cabinet", "cabinet", "desk"],
        "grocery": ["cabinet", "cabinet", "cabinet"],
    },
    "shrine": {
        "office": ["plant_pot", "whiteboard", "lamp"],
        "dark": ["topiary", "flower", "rock"],
        "pub": ["plant_pot", "lamp", "flower"],
        "school": ["whiteboard", "plant_pot", "lamp"],
        "grocery": ["plant_pot", "plant_pot", "lamp"],
    },
    "thoroughfare": {
        "office": ["lamp", "plant_pot", "partition"],
        "dark": ["rock", "topiary", "lamp"],
        "pub": ["lamp", "plant_pot", "chair"],
        "school": ["lamp", "plant_pot", "plant_pot"],
        "grocery": ["plant_pot", "plant_pot", "lamp"],
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
    if has("pub", "inn", "bar", "alehouse", "barmaid", "innkeeper", "brew", "mead", "tavern"):
        return "pub"
    if has(
        "school",
        "classroom",
        "teacher",
        "student",
        "academy",
        "university",
        "professor",
        "homework",
        "exam",
        "library",
    ):
        return "school"
    if has(
        "grocery",
        "supermarket",
        "store",
        "aisle",
        "checkout",
        "produce",
        "deli",
        "bakery",
        "market",
        "cart",
        "shopping",
    ):
        return "grocery"
    if has(
        "account",
        "filing",
        "cabinet",
        "desk",
        "report",
        "tps",
        "office",
        "cubicle",
        "convince",
        "negotiate",
        "building manager",
        "facilities",
        "vendor",
        "hr",
        "budget",
        "approve",
        "deadline",
        "fellowship",
        "party must",
        "someone has",
        "something has",
        "must determine",
        "must deliver",
        "must escort",
        "must retrieve",
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
    return "office"


def _place_zone_props(
    rng: random.Random,
    name: str,
    theme: str,
    cx: int,
    cy: int,
    rw: int,
    rh: int,
    props: list[dict],
    occupied: set[tuple[int, int]],
) -> None:
    ix0 = cx - rw // 2 + 1
    ix1 = cx + rw // 2
    iy0 = cy - rh // 2 + 1
    iy1 = cy + rh // 2

    def p(x: int, y: int, kind: str) -> bool:
        if ix0 <= x < ix1 and iy0 <= y < iy1 and (x, y) not in occupied:
            props.append({"x": x, "y": y, "kind": kind})
            occupied.add((x, y))
            return True
        return False

    archetype = _ZONE_ARCHETYPES.get(name, "thoroughfare")
    pool = _ARCHETYPE_PROPS.get(archetype, {})
    kinds = pool.get(theme) or pool.get("office") or ["rock"]
    a, b, c = kinds[0], kinds[min(1, len(kinds) - 1)], kinds[min(2, len(kinds) - 1)]

    if archetype == "work_area":
        for xi in range(ix0 + 1, ix1 - 1, 4):
            p(xi, iy0 + 1, a)
            p(xi, iy0 + 2, b)
            p(xi, iy1 - 2, a)
            p(xi, iy1 - 3, b)
        for xi in range(ix0 + 2, ix1 - 2, 6):
            p(xi, cy, c)
        p(ix0 + 1, iy0 + 1, "plant_pot")
        p(ix1 - 2, iy0 + 1, "plant_pot")

    elif archetype == "meeting_space":
        for xi in range(ix0 + 2, ix1 - 1, 3):
            p(xi, cy, a)
            p(xi, cy - 1, b)
            p(xi, cy + 1, b)
        p(ix0 + 1, cy - 1, b)
        p(ix0 + 1, cy + 1, b)
        p(ix1 - 2, cy - 1, b)
        p(ix1 - 2, cy + 1, b)
        p(cx - 1, iy0 + 1, c)
        p(cx, iy0 + 1, c)

    elif archetype == "lounge":
        p(ix0 + 1, iy0 + 1, b)
        p(ix0 + 2, iy0 + 1, b)
        p(ix0 + 3, iy0 + 1, b)
        for dx, dy in [(-1, 0), (0, 0), (-1, 1), (0, 1)]:
            p(cx + dx, cy + dy, a)
        p(ix1 - 2, iy0 + 1, c)
        p(ix1 - 2, iy1 - 2, c)
        p(ix0 + 1, iy1 - 2, c)

    elif archetype == "entrance":
        p(cx - 1, iy1 - 2, a)
        p(cx, iy1 - 2, a)
        p(cx - 1, iy1 - 3, b)
        p(ix0 + 1, cy - 1, b)
        p(ix0 + 1, cy, b)
        p(ix0 + 1, cy + 1, b)
        p(ix0 + 1, iy0 + 1, c)
        p(ix1 - 2, iy0 + 1, c)

    elif archetype == "equipment_bay":
        for xi in range(ix0 + 2, ix1 - 3, 5):
            p(xi, cy, a)
            p(xi + 1, cy, a)
        for xi in range(ix0 + 1, ix1, 4):
            p(xi, iy0 + 1, b)
        p(ix1 - 2, iy1 - 2, c)

    elif archetype == "private_office":
        p(cx - 1, iy0 + 2, a)
        p(cx, iy0 + 2, a)
        p(cx - 1, iy0 + 3, b)
        p(cx - 2, cy, b)
        p(cx + 1, cy, b)
        p(ix1 - 2, iy0 + 1, c)
        p(ix0 + 1, iy1 - 2, "plant_pot")
        p(ix1 - 2, iy1 - 2, "plant_pot")

    elif archetype == "storage":
        for yi in range(iy0 + 1, iy1 - 1, 3):
            for xi in range(ix0 + 1, ix1 - 1, 3):
                p(xi, yi, a)
                p(xi + 1, yi, a)

    elif archetype == "shrine":
        for xi in range(ix0 + 1, ix1 - 1, 4):
            p(xi, iy0 + 1, b)
            p(xi, iy1 - 2, b)
        for yi in range(iy0 + 2, iy1 - 1, 3):
            p(ix0 + 1, yi, a)
            p(ix1 - 2, yi, a)
        p(cx - 1, cy, c)
        p(cx, cy, c)

    else:
        for xi in range(ix0 + 2, ix1 - 2, 5):
            p(xi, iy0 + 1, rng.choice(kinds))
            p(xi, iy1 - 2, rng.choice(kinds))


def _weighted_splits(total: int, weights: list[int]) -> list[int]:
    s = sum(weights) or 1
    pts: list[int] = [0]
    acc = 0.0
    for wt in weights[:-1]:
        acc += wt * total / s
        pts.append(round(acc))
    pts.append(total)
    return pts


_BULLPEN_NAMES: dict[str, list[str]] = {
    "office": ["Sales", "Accounting", "Open Plan", "The Annex", "Operations", "Customer Service"],
    "dark": ["The Pit", "Common Floor", "The Stacks", "The Hollow", "Gathering Floor"],
    "pub": ["The Floor", "Long Tables", "By the Fire", "The Snug", "Bar Floor"],
    "school": ["Study Hall", "The Commons", "Desk Rows", "The Wing", "Open Hall"],
    "grocery": ["Center Aisles", "Checkout", "The Floor", "End Caps", "Back Aisles"],
}


def generate_indoor_world(
    rng: random.Random, theme: str = "office", step_count: int = 6
) -> WorldMap:
    # A natural mid-sized office: a top row of small private rooms (with doors)
    # opening onto a single open-plan bullpen full of desk clusters below.
    pool = INDOOR_ROOM_NAMES.get(theme, INDOOR_ROOM_NAMES["office"])
    zone_def_map = _ZONE_DEF.get(theme, _ZONE_DEF["office"])
    bull_pool = _BULLPEN_NAMES.get(theme, _BULLPEN_NAMES["office"])

    n_total = max(4, min(len(pool) + len(bull_pool), step_count + 1))
    perimeter_names = [nm for nm in pool if nm != "Open Plan"]
    rng.shuffle(perimeter_names)
    n_rooms = min(len(perimeter_names), max(2, (n_total + 1) // 2))
    n_desks = max(1, n_total - n_rooms)

    ROOM_IW, ROOM_IH = 7, 4
    div_row = ROOM_IH + 1  # wall line between the rooms and the bullpen
    bull_y0 = div_row + 1

    DESK_COLS = min(n_desks, 3)
    DESK_ROWS = (n_desks + DESK_COLS - 1) // DESK_COLS
    CLUS_W, CLUS_H = 9, 6

    rooms_w = n_rooms * (ROOM_IW + 2) + 1
    desks_w = DESK_COLS * CLUS_W + 3
    w = max(rooms_w, desks_w, 30)
    bull_h = DESK_ROWS * CLUS_H + 2
    h = bull_y0 + bull_h + 1

    tiles = [HEDGE] * (w * h)
    for y in range(bull_y0, h - 1):
        for x in range(1, w - 1):
            tiles[_idx(w, x, y)] = FLOOR

    room_data: list[tuple[int, int, str, int, int, int]] = []
    waypoints: list[list[int]] = []

    # top row of small private rooms, each with a door into the bullpen
    cell_w = w / n_rooms
    for i in range(n_rooms):
        name = perimeter_names[i]
        fl = zone_def_map.get(name, (FLOOR, 18, 10))[0]
        cx = int((i + 0.5) * cell_w)
        x0 = max(1, cx - ROOM_IW // 2)
        x1 = min(w - 2, cx + ROOM_IW // 2)
        for y in range(1, ROOM_IH + 1):
            for x in range(x0, x1 + 1):
                tiles[_idx(w, x, y)] = fl
        tiles[_idx(w, cx, div_row)] = fl  # door
        cy = 1 + ROOM_IH // 2
        room_data.append((cx, cy, name, fl, x1 - x0 + 1, ROOM_IH))
        waypoints.append([cx, cy])

    # open-plan bullpen: desk clusters laid out in a grid
    inner_w = w - 4
    for j in range(n_desks):
        col = j % DESK_COLS
        row = j // DESK_COLS
        cx = 2 + int((col + 0.5) * (inner_w / DESK_COLS))
        cy = min(bull_y0 + 1 + row * CLUS_H + CLUS_H // 2, h - 3)
        name = bull_pool[j % len(bull_pool)]
        room_data.append((cx, cy, name, FLOOR, CLUS_W, CLUS_H))
        waypoints.append([cx, cy])

    occupied: set[tuple[int, int]] = {(cx, cy) for cx, cy, *_ in room_data}
    props: list[dict] = []
    for cx, cy, name, fl, rw, rh in room_data[:n_rooms]:
        _place_zone_props(rng, name, theme, cx, cy, rw, rh, props, occupied)

    def _put(x: int, y: int, kind: str) -> None:
        if 1 <= x < w - 1 and bull_y0 <= y < h - 1 and (x, y) not in occupied:
            props.append({"x": x, "y": y, "kind": kind})
            occupied.add((x, y))

    wp_set = {(cx, cy) for cx, cy, *_ in room_data}
    for cx, cy, *_ in room_data[n_rooms:]:
        _put(cx - 1, cy - 1, "desk")
        _put(cx, cy - 1, "desk")
        _put(cx - 1, cy + 1, "chair")
        _put(cx, cy + 1, "chair")
        _put(cx + 1, cy, "plant_pot")

    # fill the rest of the bullpen so it reads as a packed open-plan, not bare floor —
    # cubicle rows of desks/partitions, leaving walking lanes and clear space around NPCs
    _filler = ["desk", "desk", "partition", "cabinet", "chair", "plant_pot"]
    for gy in range(bull_y0 + 1, h - 2, 2):
        for gx in range(2, w - 2, 3):
            if any((gx + ox, gy + oy) in wp_set for ox in (-1, 0, 1) for oy in (-1, 0, 1)):
                continue  # keep space clear around the people you walk up to
            if (gx, gy) in occupied:
                continue
            if rng.random() < 0.62:
                _put(gx, gy, rng.choice(_filler))

    indoor_walkable = {FLOOR, PATH, PATH_LIGHT, LAVENDER}
    route: list[list[int]] = []
    wp_route_idx: list[int] = []
    for i, (cx, cy, *_) in enumerate(room_data):
        if i == 0:
            wp_route_idx.append(0)
            route.append([cx, cy])
            continue
        prev = room_data[i - 1]
        seg = _bfs(w, h, tiles, [prev[0], prev[1]], [cx, cy], walkable=indoor_walkable)
        wp_route_idx.append(len(route) - 1 + len(seg) - 1)
        route.extend(seg[1:])

    frames = INDOOR_TILE_FRAMES.get(theme, INDOOR_TILE_FRAMES["office"])
    return WorldMap(
        w=w,
        h=h,
        tiles=tiles,
        props=props,
        waypoints=waypoints,
        waypoint_names=[rd[2] for rd in room_data],
        route=route,
        wp_route_idx=wp_route_idx,
        theme=theme,
        tint=THEMES.get(theme, THEMES["office"])["tint"],
        tileset="interior",
        tile_frames={str(k): v for k, v in frames.items()},
        walkable=sorted(indoor_walkable),
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
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = wx + dx, wy + dy
                if 1 <= nx < w - 1 and 1 <= ny < h - 1:
                    occupied.add((nx, ny))

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
        walkable=sorted(OUTDOOR_WALKABLE),
    )


def generate_world(rng: random.Random, theme: str = "garden", step_count: int = 6) -> WorldMap:
    if theme in INDOOR_THEMES:
        return generate_indoor_world(rng, theme=theme, step_count=step_count)
    return generate_outdoor_world(rng, theme=theme, step_count=step_count)


def facing_from_delta(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"
