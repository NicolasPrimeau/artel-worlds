from __future__ import annotations

# Force a give-and-go in the FINAL THIRD (where it matters) and render it, to judge the actuator's
# quality in a good spot — independent of the dumb stand-in trigger.

from PIL import Image

from pitch.engine import Pitch
from pitch.plays import PlayManager, play_brain
from tools.play_demo import draw


def main():
    p = Pitch(seed=5)
    p.setup(["x"] * 9, ["y"] * 9)
    home = [pl for pl in p.players if pl.team == "home"]
    away = [pl for pl in p.players if pl.team == "away"]
    carrier = next(pl for pl in home if pl.role == "FWD")
    wall = next(pl for pl in home if pl.role == "MID")
    gk = next(pl for pl in away if pl.role == "GK")
    defs = [pl for pl in away if pl.role == "DEF"][:3]
    presser = next(pl for pl in away if pl.role == "MID")
    carrier.x, carrier.y = 84, 44
    wall.x, wall.y = 80, 33
    presser.x, presser.y = 90, 43  # the man to beat with the one-two
    for d, y in zip(defs, (26, 40, 54)):
        d.x, d.y = 100, y  # a flat back line further behind
    gk.x, gk.y = 114, 40
    for o in away:  # push everyone else out of the lane
        if o not in (gk, presser, *defs):
            o.x, o.y = min(o.x, 95), o.y
    p.ball.x, p.ball.y = carrier.x, carrier.y
    p.possessor = carrier.id

    mgr = PlayManager("home")
    brain = play_brain("home", mgr)
    frames = []
    for _ in range(30):
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
    print("history:", [(s, e, o) for s, e, _, o in mgr.history])
    hl = {carrier.id: (255, 235, 0), wall.id: (0, 230, 230)}
    imgs = [draw(p.cfg, f, hl)[0] for f in frames]
    imgs[0].save("/tmp/gg_final.gif", save_all=True, append_images=imgs[1:], duration=140, loop=0)
    keys = list(range(0, len(frames), max(1, len(frames) // 8)))[:8]
    tiles = [draw(p.cfg, frames[i], hl) for i in keys]
    tw, th = tiles[0][1]
    tw, th = tw // 2, th // 2
    montage = Image.new("RGB", (tw * 2, th * 4), (0, 0, 0))
    for i, (im, _) in enumerate(tiles):
        montage.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
    montage.save("/tmp/gg_final_montage.png")
    print("wrote /tmp/gg_final.gif, /tmp/gg_final_montage.png")


if __name__ == "__main__":
    main()
