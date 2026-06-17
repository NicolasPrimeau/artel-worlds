from PIL import Image
from pitch.engine import Pitch
from pitch.commander import Coordinator, Plan, combined_brain
from tools.play_demo import draw

co = Coordinator("home")
co.plan = Plan(58.0, 1, False, True)
co.plays.call = {"combos": True, "channel": "right"}
brain = combined_brain({"home": co})
p = Pitch(seed=7); p.setup(["x"] * 9, ["y"] * 9)
frames = []
for _ in range(1200):
    p.step(brain)
    pl = co.plays.play
    frames.append({"ball": (p.ball.x, p.ball.y), "score": dict(p.score), "phase": pl.phase if pl else None,
                   "players": [(q.id, q.x, q.y, q.team, q.role, q.number) for q in p.players]})
done = [(s, e, pr) for s, e, pr, o in co.plays.history if o == "completed"]
print("started", co.plays.started, "completed", co.plays.completed, "score", dict(p.score))
if done:
    st, en, (c2, w2) = done[0]
    s, e = max(0, st - 1 - 3), min(len(frames), en - 1 + 8)
    hl = {c2: (255, 235, 0), w2: (0, 230, 230)}
    keys = list(range(s, e, max(1, (e - s) // 8)))[:8]
    tiles = [draw(p.cfg, frames[i], hl) for i in keys]
    tw, th = tiles[0][1]; tw, th = tw // 2, th // 2
    m = Image.new("RGB", (tw * 2, th * 4), (0, 0, 0))
    for i, (im, _) in enumerate(tiles): m.paste(im.resize((tw, th)), ((i % 2) * tw, (i // 2) * th))
    m.save("/tmp/gg_integrated.png"); print("wrote /tmp/gg_integrated.png ticks", st, en)
