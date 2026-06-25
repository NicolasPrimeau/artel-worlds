#!/usr/bin/env python3
"""Generate 10 office worker sprite sheets (64x112, 16x16 tiles, 4 dir x 7 rows)."""

from PIL import Image, ImageDraw

OUTPUT_DIR = "/home/nprimeau/projects/artel-worlds/vibequest/static/assets"

CHARACTERS = [
    {
        "name": "hero_01",
        "skin": (255, 214, 170),
        "hair": (30, 20, 10),
        "shirt": (30, 50, 100),
        "pants": (20, 20, 30),
        "shoe": (35, 22, 8),
        "tie": (140, 25, 25),
        "long": False,
    },
    {
        "name": "hero_02",
        "skin": (210, 168, 120),
        "hair": (85, 48, 12),
        "shirt": (60, 60, 60),
        "pants": (45, 45, 55),
        "shoe": (22, 14, 5),
        "tie": None,
        "long": True,
    },
    {
        "name": "hero_03",
        "skin": (155, 105, 65),
        "hair": (18, 12, 5),
        "shirt": (95, 28, 38),
        "pants": (28, 22, 42),
        "shoe": (18, 10, 4),
        "tie": None,
        "long": False,
    },
    {
        "name": "hero_04",
        "skin": (100, 62, 32),
        "hair": (22, 15, 5),
        "shirt": (52, 72, 42),
        "pants": (38, 28, 18),
        "shoe": (22, 14, 5),
        "tie": (90, 65, 18),
        "long": False,
    },
    {
        "name": "hero_05",
        "skin": (242, 200, 158),
        "hair": (182, 142, 95),
        "shirt": (72, 92, 122),
        "pants": (30, 30, 52),
        "shoe": (28, 18, 8),
        "tie": (108, 32, 52),
        "long": True,
    },
    {
        "name": "hero_06",
        "skin": (252, 212, 172),
        "hair": (125, 125, 125),
        "shirt": (28, 82, 82),
        "pants": (32, 32, 32),
        "shoe": (18, 12, 4),
        "tie": None,
        "long": False,
    },
    {
        "name": "hero_07",
        "skin": (188, 138, 98),
        "hair": (65, 32, 12),
        "shirt": (62, 42, 28),
        "pants": (38, 28, 16),
        "shoe": (18, 10, 4),
        "tie": (42, 62, 82),
        "long": True,
    },
    {
        "name": "hero_08",
        "skin": (78, 48, 22),
        "hair": (12, 8, 3),
        "shirt": (18, 18, 22),
        "pants": (28, 28, 32),
        "shoe": (14, 8, 2),
        "tie": (155, 52, 22),
        "long": False,
    },
    {
        "name": "hero_09",
        "skin": (245, 208, 168),
        "hair": (52, 28, 6),
        "shirt": (48, 58, 68),
        "pants": (42, 42, 42),
        "shoe": (30, 20, 10),
        "tie": None,
        "long": True,
    },
    {
        "name": "hero_10",
        "skin": (128, 82, 48),
        "hair": (28, 18, 6),
        "shirt": (22, 58, 32),
        "pants": (28, 28, 38),
        "shoe": (18, 10, 4),
        "tie": (85, 22, 42),
        "long": False,
    },
]

TRANSPARENT = (0, 0, 0, 0)


def c(color):
    return color + (255,) if len(color) == 3 else color


def px(draw, ox, oy, x, y, color):
    if 0 <= x <= 15 and 0 <= y <= 15:
        draw.point((ox + x, oy + y), fill=c(color))


def row(draw, ox, oy, y, x1, x2, color):
    for x in range(x1, x2 + 1):
        px(draw, ox, oy, x, y, color)


# ── Front (facing down, col 0) ────────────────────────────────────────────────


def head_front(draw, ox, oy, cfg):
    H, S, E = cfg["hair"], cfg["skin"], (28, 18, 8)
    long = cfg["long"]
    # hair top
    row(draw, ox, oy, 1, 4, 11, H)
    row(draw, ox, oy, 2, 3, 12, H)
    row(draw, ox, oy, 3, 3, 12, H)
    # face rows
    face_top = 4 if not long else 5
    row(draw, ox, oy, face_top, 3, 12, H)  # hair sides + face
    for x in range(5, 11):
        px(draw, ox, oy, x, face_top, S)  # skin in center
    row(draw, ox, oy, face_top + 1, 5, 10, S)  # eye row
    px(draw, ox, oy, 3, face_top + 1, H)
    px(draw, ox, oy, 12, face_top + 1, H)
    px(draw, ox, oy, 6, face_top + 1, E)  # left eye
    px(draw, ox, oy, 9, face_top + 1, E)  # right eye
    row(draw, ox, oy, face_top + 2, 5, 10, S)  # lower face
    row(draw, ox, oy, face_top + 3, 5, 10, S)  # chin
    if not long:
        row(draw, ox, oy, 8, 6, 9, S)  # neck — bridges gap to collar for short hair


def body_front(draw, ox, oy, cfg, frame):
    T, P = cfg["shirt"], cfg["pants"]
    Ti = cfg["tie"]
    # collar
    row(draw, ox, oy, 9, 5, 10, T)
    # shirt
    for y in range(10, 13):
        row(draw, ox, oy, y, 4, 11, T)
    # shoulders wider
    px(draw, ox, oy, 3, 10, T)
    px(draw, ox, oy, 12, 10, T)
    # belt/hips
    row(draw, ox, oy, 13, 4, 11, P)
    # tie (center, down-facing)
    if Ti:
        px(draw, ox, oy, 8, 10, Ti)
        px(draw, ox, oy, 8, 11, Ti)
        px(draw, ox, oy, 7, 11, Ti)
        px(draw, ox, oy, 8, 12, Ti)


def legs_front(draw, ox, oy, cfg, frame):
    P, S = cfg["pants"], cfg["shoe"]
    # leg configs per frame: (left_x, right_x, left_extend, right_extend)
    configs = [
        (5, 9, 0, 0),  # frame 0: neutral
        (4, 10, 1, 0),  # frame 1: left forward
        (6, 8, 0, 0),  # frame 2: crossing
        (4, 10, 0, 1),  # frame 3: right forward
    ]
    lx, rx, le, re = configs[frame % 4]
    for y in range(14, 15):
        px(draw, ox, oy, lx, y, P)
        px(draw, ox, oy, lx + 1, y, P)
        px(draw, ox, oy, rx, y, P)
        px(draw, ox, oy, rx + 1, y, P)
    # shoe row
    px(draw, ox, oy, lx, 15, S)
    px(draw, ox, oy, lx + 1, 15, S)
    if le:
        px(draw, ox, oy, lx - 1, 15, S)
    px(draw, ox, oy, rx, 15, S)
    px(draw, ox, oy, rx + 1, 15, S)
    if re:
        px(draw, ox, oy, rx + 2, 15, S)


# ── Back (facing up, col 1) ───────────────────────────────────────────────────


def head_back(draw, ox, oy, cfg):
    H, S = cfg["hair"], cfg["skin"]
    long = cfg["long"]
    row(draw, ox, oy, 1, 4, 11, H)
    row(draw, ox, oy, 2, 3, 12, H)
    row(draw, ox, oy, 3, 3, 12, H)
    row(draw, ox, oy, 4, 3, 12, H)
    if long:
        row(draw, ox, oy, 5, 3, 12, H)
        row(draw, ox, oy, 6, 4, 11, H)
        row(draw, ox, oy, 7, 5, 10, H)
        # neck visible below long hair
        row(draw, ox, oy, 8, 6, 9, S)
    else:
        row(draw, ox, oy, 5, 4, 11, H)
        # nape + neck
        row(draw, ox, oy, 6, 5, 10, H)
        row(draw, ox, oy, 7, 6, 9, S)
        row(draw, ox, oy, 8, 6, 9, S)


def body_back(draw, ox, oy, cfg, frame):
    T = cfg["shirt"]
    # back of shirt (no tie, wider)
    row(draw, ox, oy, 9, 5, 10, T)
    px(draw, ox, oy, 3, 9, T)
    px(draw, ox, oy, 12, 9, T)
    for y in range(10, 14):
        row(draw, ox, oy, y, 4, 11, T)


def legs_back(draw, ox, oy, cfg, frame):
    legs_front(draw, ox, oy, cfg, frame)


# ── Left profile (col 2) ──────────────────────────────────────────────────────


def head_left(draw, ox, oy, cfg):
    H, S, E = cfg["hair"], cfg["skin"], (28, 18, 8)
    # Profile: head is to the right side, hair on left
    row(draw, ox, oy, 1, 5, 10, H)
    row(draw, ox, oy, 2, 4, 11, H)
    # face area: left side hair, right side skin
    for y in range(3, 7):
        px(draw, ox, oy, 4, y, H)
        px(draw, ox, oy, 5, y, H)
        for x in range(6, 11):
            px(draw, ox, oy, x, y, S)
        px(draw, ox, oy, 11, y, H)
    # eye visible on profile (right eye, left-facing)
    px(draw, ox, oy, 9, 4, E)
    # chin point
    row(draw, ox, oy, 7, 6, 9, S)


def body_left(draw, ox, oy, cfg, frame):
    T, P = cfg["shirt"], cfg["pants"]
    # Narrower side view
    row(draw, ox, oy, 8, 6, 10, T)
    for y in range(9, 13):
        row(draw, ox, oy, y, 6, 10, T)
    row(draw, ox, oy, 13, 6, 10, P)


def legs_left(draw, ox, oy, cfg, frame):
    P, S = cfg["pants"], cfg["shoe"]
    # Side-view legs: front and back leg
    # frame 0/2: neutral stacked
    # frame 1: front (right, toward left) forward
    # frame 3: back (left) forward
    if frame % 2 == 0:
        for y in range(14, 15):
            row(draw, ox, oy, y, 6, 9, P)
        row(draw, ox, oy, 15, 6, 9, S)
    elif frame == 1:
        # front leg steps left (lower x)
        for y in range(14, 15):
            row(draw, ox, oy, y, 5, 8, P)  # front
            row(draw, ox, oy, y, 9, 10, P)  # back
        row(draw, ox, oy, 15, 4, 7, S)  # front shoe
        row(draw, ox, oy, 15, 9, 10, S)  # back shoe
    else:  # frame 3
        for y in range(14, 15):
            row(draw, ox, oy, y, 5, 8, P)
            row(draw, ox, oy, y, 9, 10, P)
        row(draw, ox, oy, 15, 9, 11, S)  # front shoe (right = back direction)
        row(draw, ox, oy, 15, 5, 7, S)  # back shoe


# ── Right profile (col 3): mirror of left ─────────────────────────────────────


def head_right(draw, ox, oy, cfg):
    H, S, E = cfg["hair"], cfg["skin"], (28, 18, 8)
    row(draw, ox, oy, 1, 5, 10, H)
    row(draw, ox, oy, 2, 4, 11, H)
    for y in range(3, 7):
        px(draw, ox, oy, 11, y, H)
        px(draw, ox, oy, 10, y, H)
        for x in range(5, 10):
            px(draw, ox, oy, x, y, S)
        px(draw, ox, oy, 4, y, H)
    px(draw, ox, oy, 6, 4, E)
    row(draw, ox, oy, 7, 6, 9, S)


def body_right(draw, ox, oy, cfg, frame):
    body_left(draw, ox, oy, cfg, frame)


def legs_right(draw, ox, oy, cfg, frame):
    P, S = cfg["pants"], cfg["shoe"]
    if frame % 2 == 0:
        for y in range(14, 15):
            row(draw, ox, oy, y, 6, 9, P)
        row(draw, ox, oy, 15, 6, 9, S)
    elif frame == 1:
        for y in range(14, 15):
            row(draw, ox, oy, y, 7, 10, P)
            row(draw, ox, oy, y, 5, 6, P)
        row(draw, ox, oy, 15, 8, 11, S)
        row(draw, ox, oy, 15, 5, 6, S)
    else:
        for y in range(14, 15):
            row(draw, ox, oy, y, 7, 10, P)
            row(draw, ox, oy, y, 5, 6, P)
        row(draw, ox, oy, 15, 5, 7, S)
        row(draw, ox, oy, 15, 9, 10, S)


# ── Tile dispatcher ───────────────────────────────────────────────────────────


def draw_tile(draw, col, row_idx, cfg):
    ox = col * 16
    oy = row_idx * 16
    frame = row_idx % 4

    if col == 0:
        head_front(draw, ox, oy, cfg)
        body_front(draw, ox, oy, cfg, frame)
        legs_front(draw, ox, oy, cfg, frame)
    elif col == 1:
        head_back(draw, ox, oy, cfg)
        body_back(draw, ox, oy, cfg, frame)
        legs_back(draw, ox, oy, cfg, frame)
    elif col == 2:
        head_right(draw, ox, oy, cfg)  # head_right has face on left side = walking left
        body_left(draw, ox, oy, cfg, frame)
        legs_left(draw, ox, oy, cfg, frame)
    elif col == 3:
        head_left(draw, ox, oy, cfg)  # head_left has face on right side = walking right
        body_right(draw, ox, oy, cfg, frame)
        legs_right(draw, ox, oy, cfg, frame)


def make_sheet(cfg):
    img = Image.new("RGBA", (64, 112), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    for col in range(4):
        for row_idx in range(7):
            draw_tile(draw, col, row_idx, cfg)
    return img


def main():
    import os

    for cfg in CHARACTERS:
        path = os.path.join(OUTPUT_DIR, f"{cfg['name']}.png")
        make_sheet(cfg).save(path, "PNG")
        print(f"  {path}")
    print(f"Done — {len(CHARACTERS)} sheets")


if __name__ == "__main__":
    main()
