from __future__ import annotations

# Dev-only match renderer (not shipped). Run a match headless, draw the positions to a GIF so we can
# actually WATCH behaviour — pressing shape, runs, passing, bunching — instead of trusting metrics.
# Usage: uv run python tools/render.py            -> baseline match
#        renders a gif + a key-frame montage png under /tmp.

from PIL import Image, ImageDraw

from pitch.bot import decide
from pitch.engine import Pitch

SCALE = 7  # px per metre
PAD = 16
TEAM = {"home": (60, 130, 240), "away": (235, 70, 70)}  # blue / red
GK = {"home": (150, 190, 255), "away": (255, 160, 160)}


def _xy(x: float, y: float, c) -> tuple[float, float]:
    return PAD + x * SCALE, PAD + y * SCALE


def record(brain, ticks: int, seed: int = 7) -> tuple:
    p = Pitch(seed=seed)
    p.setup(["x"] * 9, ["y"] * 9)
    frames = []
    for _ in range(ticks):
        p.step(brain)
        frames.append(
            {
                "ball": (p.ball.x, p.ball.y),
                "poss": p.possessor,
                "players": [(pl.x, pl.y, pl.team, pl.role, pl.number) for pl in p.players],
                "restart": p.restart_kind,
                "score": dict(p.score),
            }
        )
    return p.cfg, frames


def draw(cfg, f) -> Image.Image:
    W = int(PAD * 2 + cfg.length * SCALE)
    H = int(PAD * 2 + cfg.width * SCALE)
    img = Image.new("RGB", (W, H), (28, 120, 52))
    d = ImageDraw.Draw(img)
    line = (255, 255, 255)
    d.rectangle([_xy(0, 0, cfg), _xy(cfg.length, cfg.width, cfg)], outline=line, width=2)
    d.line([_xy(cfg.length / 2, 0, cfg), _xy(cfg.length / 2, cfg.width, cfg)], fill=line, width=2)
    cx, cy = _xy(cfg.length / 2, cfg.width / 2, cfg)
    d.ellipse(
        [cx - 9 * SCALE, cy - 9 * SCALE, cx + 9 * SCALE, cy + 9 * SCALE], outline=line, width=2
    )
    for gx in (0, cfg.length):  # goals + boxes
        d.rectangle(
            [
                _xy(gx if gx == 0 else gx - 16, cfg.width / 2 - 20, cfg),
                _xy(gx + 16 if gx == 0 else gx, cfg.width / 2 + 20, cfg),
            ],
            outline=line,
            width=1,
        )
    for x, y, team, role, num in f["players"]:
        px, py = _xy(x, y, cfg)
        r = 6 * SCALE / 7
        col = (GK if role == "GK" else TEAM)[team]
        d.ellipse([px - r, py - r, px + r, py + r], fill=col, outline=(0, 0, 0))
        d.text((px - 3, py - 5), str(num), fill=(0, 0, 0))
    bx, by = _xy(*f["ball"], cfg)
    d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255), outline=(0, 0, 0))
    tag = f["restart"] or ("GOAL!" if False else "")
    d.text((PAD, 2), f"{f['score']['home']}-{f['score']['away']}  {tag}", fill=(255, 255, 0))
    return img


def main() -> None:
    cfg, frames = record(decide, ticks=420)  # ~34s of play
    imgs = [draw(cfg, frames[i]) for i in range(0, len(frames), 3)]  # ~12 fps
    gif = "/tmp/pitch_baseline.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=80, loop=0)
    # a montage of 8 key frames so a single still shows a sequence
    keys = [frames[i] for i in range(0, len(frames), max(1, len(frames) // 8))][:8]
    tiles = [
        draw(cfg, k).resize(
            (cfg and 0)
            or (int((PAD * 2 + cfg.length * SCALE) * 0.5), int((PAD * 2 + cfg.width * SCALE) * 0.5))
        )
        for k in keys
    ]
    tw, th = tiles[0].size
    montage = Image.new("RGB", (tw * 2, th * 4), (0, 0, 0))
    for i, t in enumerate(tiles):
        montage.paste(t, ((i % 2) * tw, (i // 2) * th))
    montage.save("/tmp/pitch_montage.png")
    print("wrote", gif, "and /tmp/pitch_montage.png |", len(imgs), "frames")


if __name__ == "__main__":
    main()
