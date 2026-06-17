from __future__ import annotations

# Render a coordinated play being CARRIED OUT, to verify the actuators follow through. Home runs the
# play layer (heuristic stand-in for the LLM brain); away is baseline. We find a completed give-and-go
# and render the window around it, ringing the two players involved (carrier=yellow, wall=cyan).

from PIL import Image, ImageDraw

from pitch.engine import Pitch
from pitch.plays import PlayManager, play_brain

SCALE = 7
PAD = 16
TEAM = {"home": (60, 130, 240), "away": (235, 70, 70)}
GK = {"home": (150, 190, 255), "away": (255, 160, 160)}


def _xy(x, y):
    return PAD + x * SCALE, PAD + y * SCALE


def draw(cfg, f, hl):
    W = int(PAD * 2 + cfg.length * SCALE)
    H = int(PAD * 2 + cfg.width * SCALE)
    img = Image.new("RGB", (W, H), (28, 120, 52))
    d = ImageDraw.Draw(img)
    line = (255, 255, 255)
    d.rectangle([_xy(0, 0), _xy(cfg.length, cfg.width)], outline=line, width=2)
    d.line([_xy(cfg.length / 2, 0), _xy(cfg.length / 2, cfg.width)], fill=line, width=2)
    cx, cy = _xy(cfg.length / 2, cfg.width / 2)
    d.ellipse([cx - 9 * SCALE, cy - 9 * SCALE, cx + 9 * SCALE, cy + 9 * SCALE], outline=line, width=2)
    for pid, x, y, team, role, num in f["players"]:
        px, py = _xy(x, y)
        r = 6 * SCALE / 7
        if pid in hl:
            d.ellipse([px - r - 5, py - r - 5, px + r + 5, py + r + 5], outline=hl[pid], width=3)
        d.ellipse([px - r, py - r, px + r, py + r], fill=(GK if role == "GK" else TEAM)[team], outline=(0, 0, 0))
        d.text((px - 3, py - 5), str(num), fill=(0, 0, 0))
    bx, by = _xy(*f["ball"])
    d.ellipse([bx - 4, by - 4, bx + 4, by + 4], fill=(255, 255, 255), outline=(0, 0, 0))
    d.text((PAD, 2), f"{f['score']['home']}-{f['score']['away']}  {f.get('phase') or ''}", fill=(255, 255, 0))
    return img, (W, H)


def main():
    p = Pitch(seed=7)
    p.setup(["x"] * 9, ["y"] * 9)
    mgr = PlayManager("home")
    brain = play_brain("home", mgr)
    frames = []
    for _ in range(2400):
        p.step(brain)
        play = mgr.play
        frames.append(
            {
                "ball": (p.ball.x, p.ball.y),
                "players": [(pl.id, pl.x, pl.y, pl.team, pl.role, pl.number) for pl in p.players],
                "score": dict(p.score),
                "phase": play.phase if play else None,
            }
        )
    done = [(s, e, pl) for s, e, pl, o in mgr.history if o == "completed"]
    print(f"give-and-go: started {mgr.started}, completed {mgr.completed}")
    if not done:
        return
    st, en, (carrier, wall) = done[0]
    s, e = max(0, st - 1 - 3), min(len(frames), en - 1 + 8)  # frames[i] is tick i+1
    hl = {carrier: (255, 235, 0), wall: (0, 230, 230)}
    imgs = [draw(p.cfg, frames[i], hl)[0] for i in range(s, e)]
    imgs[0].save("/tmp/gg.gif", save_all=True, append_images=imgs[1:], duration=130, loop=0)
    keys = list(range(s, e, max(1, (e - s) // 8)))[:8]
    tiles = [draw(p.cfg, frames[i], hl) for i in keys]
    (tw, th) = tiles[0][1]
    tw, th = tw // 2, th // 2
    montage = Image.new("RGB", (tw * 2, th * 4), (0, 0, 0))
    for i, (im, _) in enumerate(tiles):
        montage.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
    montage.save("/tmp/gg_montage.png")
    print(f"rendered give-and-go ticks {st}..{en} (#{p.players[carrier].number} & #{p.players[wall].number}) -> /tmp/gg.gif, /tmp/gg_montage.png")


if __name__ == "__main__":
    main()
