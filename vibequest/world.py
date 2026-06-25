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

    p(cx, cy - 1, "lamp")

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


def generate_indoor_world(
    rng: random.Random, theme: str = "office", step_count: int = 6
) -> WorldMap:
    zone_names = INDOOR_ROOM_NAMES.get(theme, INDOOR_ROOM_NAMES["office"])
    zone_def_map = _ZONE_DEF.get(theme, _ZONE_DEF["office"])

    n = min(len(zone_names), step_count + 1)
    shuffled = zone_names.copy()
    rng.shuffle(shuffled)
    used_names = shuffled[:n]

    LEFT_CX, RIGHT_CX = 16, 48
    ROW_PITCH = 18
    MARGIN = 6
    n_rows = (n + 1) // 2

    w = 64
    h = n_rows * ROW_PITCH + MARGIN * 2
    tiles = [HEDGE] * (w * h)

    room_data: list[tuple[int, int, str, int, int, int]] = []
    waypoints: list[list[int]] = []
    for i in range(n):
        row = i // 2
        col = i % 2
        cx = LEFT_CX if col == 0 else RIGHT_CX
        cy = h - MARGIN - ROW_PITCH // 2 - row * ROW_PITCH
        name = used_names[i]
        fl, rw_r, rh_r = zone_def_map.get(name, (FLOOR, 18, 10))
        rh_r = min(rh_r, ROW_PITCH - 6)
        room_data.append((cx, cy, name, fl, rw_r, rh_r))
        waypoints.append([cx, cy])

    for cx, cy, _name, fl, rw_r, rh_r in room_data:
        hw, hh = rw_r // 2, rh_r // 2
        for y in range(cy - hh + 1, cy + hh):
            for x in range(cx - hw + 1, cx + hw):
                if 0 <= x < w and 0 <= y < h:
                    tiles[_idx(w, x, y)] = fl
        for x in range(cx - hw, cx + hw + 1):
            for bdy in (cy - hh, cy + hh):
                if 0 <= x < w and 0 <= bdy < h:
                    tiles[_idx(w, x, bdy)] = HEDGE
        for y in range(cy - hh, cy + hh + 1):
            for bdx in (cx - hw, cx + hw):
                if 0 <= bdx < w and 0 <= y < h:
                    tiles[_idx(w, bdx, y)] = HEDGE

    for row in range(n_rows - 1):
        lower_cy = h - MARGIN - ROW_PITCH // 2 - row * ROW_PITCH
        upper_cy = lower_cy - ROW_PITCH
        lower_rooms = [rd for rd in room_data if rd[1] == lower_cy]
        upper_rooms = [rd for rd in room_data if rd[1] == upper_cy]
        if not lower_rooms or not upper_rooms:
            continue
        max_rh_lower = max(rd[5] for rd in lower_rooms)
        max_rh_upper = max(rd[5] for rd in upper_rooms)
        gap_top = upper_cy + max_rh_upper // 2 + 1
        gap_bot = lower_cy - max_rh_lower // 2 - 1
        for y in range(gap_top, gap_bot + 1):
            for x in range(2, w - 2):
                tiles[_idx(w, x, y)] = PATH

    for i in range(0, n - 1, 2):
        if i + 1 >= n:
            break
        lcx, lcy, _ln, _lfl, lrw, _lrh = room_data[i]
        rcx, rcy, _rn, _rfl, rrw, _rrh = room_data[i + 1]
        lwall = lcx + lrw // 2
        rwall = rcx - rrw // 2
        for x in range(lwall + 1, rwall):
            for dy in range(-1, 2):
                ny = lcy + dy
                if 0 <= x < w and 0 <= ny < h:
                    tiles[_idx(w, x, ny)] = PATH

    for i, (cx, cy, _name, fl, rw_r, rh_r) in enumerate(room_data):
        row = i // 2
        col = i % 2
        hw, hh = rw_r // 2, rh_r // 2
        if row < n_rows - 1:
            for dx in range(-1, 2):
                if 0 <= cx + dx < w and 0 <= cy - hh < h:
                    tiles[_idx(w, cx + dx, cy - hh)] = fl
        if row > 0 or i == 0:
            for dx in range(-1, 2):
                if 0 <= cx + dx < w and 0 <= cy + hh < h:
                    tiles[_idx(w, cx + dx, cy + hh)] = fl
        if col == 0 and i + 1 < n:
            for dy in range(-1, 2):
                if 0 <= cx + hw < w and 0 <= cy + dy < h:
                    tiles[_idx(w, cx + hw, cy + dy)] = fl
        if col == 1:
            for dy in range(-1, 2):
                if 0 <= cx - hw < w and 0 <= cy + dy < h:
                    tiles[_idx(w, cx - hw, cy + dy)] = fl

    cx0, cy0, _n0, fl0, _rw0, rh0 = room_data[0]
    for y in range(cy0 + rh0 // 2 + 1, min(h - 1, cy0 + rh0 // 2 + MARGIN)):
        for dx in range(-1, 2):
            nx = cx0 + dx
            if 0 <= nx < w and 0 <= y < h:
                tiles[_idx(w, nx, y)] = PATH

    for cx, cy, *_ in room_data:
        tiles[_idx(w, cx, cy)] = LAVENDER

    occupied: set[tuple[int, int]] = {(cx, cy) for cx, cy, *_ in room_data}
    props: list[dict] = []
    for cx, cy, name, fl, rw_r, rh_r in room_data:
        _place_zone_props(rng, name, theme, cx, cy, rw_r, rh_r, props, occupied)

    indoor_walkable = {FLOOR, PATH, PATH_LIGHT, LAVENDER}
    route: list[list[int]] = []
    wp_route_idx: list[int] = []
    for i, (cx, cy, *_) in enumerate(room_data):
        if i == 0:
            wp_route_idx.append(0)
            route.append([cx, cy])
            continue
        prev_cx, prev_cy = room_data[i - 1][0], room_data[i - 1][1]
        seg = _bfs(w, h, tiles, [prev_cx, prev_cy], [cx, cy], walkable=indoor_walkable)
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
        tint=THEMES.get(theme, THEMES["office"])["tint"],
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
