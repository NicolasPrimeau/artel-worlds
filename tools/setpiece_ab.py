from __future__ import annotations

# Theory 1: set pieces are where an LLM could beat a rule (slow, open-ended, coordinated, reads the
# defence). But FIRST establish there's structure to exploit: if picking the OPEN zone (where the
# defence ISN'T) beats a fixed routine, then reading the setup matters and an LLM has a job. If a
# fixed routine is just as good, no brain helps. We force corners with a random defensive cluster and
# run each routine through the delivery, scoring whether the attack gets a clean shot.

import random
import sys

from pitch import bot
from pitch.engine import Pitch, _len

BANDS = {"top": 0.30, "center": 0.50, "bottom": 0.70}  # y-fraction of the target in the box


def make_corner(seed):
    p = Pitch(seed=seed)
    p.setup(["x"] * 9, ["y"] * 9)
    rng = random.Random(seed * 7 + 1)
    L, W = p.cfg.length, p.cfg.width
    home = [pl for pl in p.players if pl.team == "home"]
    away = [pl for pl in p.players if pl.team == "away"]
    corner_y = 1.0 if rng.random() < 0.5 else W - 1.0
    taker = home[1]
    taker.x, taker.y = L - 2.0, corner_y
    p.ball.x, p.ball.y = L - 2.0, corner_y
    p.possessor = taker.id
    gk = next(pl for pl in away if pl.role == "GK")
    gk.x, gk.y = L - 3.0, W / 2
    # defenders cluster in ONE random band of the box — the setup to read
    packed = rng.choice(list(BANDS))
    defs = [pl for pl in away if pl.role != "GK"]
    for i, dfn in enumerate(defs):
        dfn.x = L - 10.0 + (i % 3) * 2.0
        dfn.y = W * BANDS[packed] + (i - len(defs) / 2) * 1.5
    # attackers spread across the three bands at the top of the box
    atk = [pl for pl in home if pl != taker and pl.role != "GK"]
    for i, a in enumerate(atk):
        a.x = L - 13.0
        a.y = W * (0.25 + 0.5 * (i / max(1, len(atk) - 1)))
    return p, taker, packed


def run(seed, choose_band):
    p, taker, packed = make_corner(seed)
    L, W = p.cfg.length, p.cfg.width
    band = choose_band(packed)
    tx, ty = L - 9.0, W * BANDS[band]
    home = [pl for pl in p.players if pl.team == "home"]
    target = min(
        (a for a in home if a != taker and a.role != "GK"), key=lambda a: _len(a.x - tx, a.y - ty)
    )
    delivered = {"v": False}
    shot = {"v": False}

    def brain(pitch, pl):
        if pl.id == taker.id and pitch.possessor == taker.id and not delivered["v"]:
            delivered["v"] = True  # whip the corner into the chosen zone
            from pitch.engine import _unit

            ux, uy = _unit(tx - pl.x, ty - pl.y)
            return {
                "move": (tx, ty),
                "kick": (ux * pitch.cfg.pass_speed * 1.2, uy * pitch.cfg.pass_speed * 1.2),
            }
        if pl.id == target.id and not (pitch.possessor == target.id):
            return {"move": (tx, ty), "sprint": True}  # attack the delivery
        return bot.decide(pitch, pl)

    start = p.score["home"]
    for _ in range(40):
        p.step(brain)
        # a "clean shot": a home attacker strikes from inside the box toward goal
        if pitch_shot(p):
            shot["v"] = True
    return shot["v"], p.score["home"] > start


def pitch_shot(p):
    poss = p.possessor
    if poss is None:
        return False
    pl = p.players[poss]
    if pl.team != "home":
        return False
    gx = p.cfg.length
    return (gx - pl.x) < 18 and abs(pl.y - p.cfg.width / 2) < 22


def trial(label, choose_band, n=120):
    shots = goals = 0
    for s in range(n):
        sh, gl = run(s, choose_band)
        shots += sh
        goals += gl
        print(f"\r{label}: {s + 1}/{n}", end="", file=sys.stderr)
    print(f"\r{label:<26} shot {round(shots / n * 100)}% | goal {round(goals / n * 100)}% (n={n})")


def open_band(packed):
    # read the setup: attack the off-keeper EDGE (top/bottom) that the defenders did NOT pack
    return "bottom" if packed == "top" else "top"


if __name__ == "__main__":
    trial("fixed center", lambda packed: "center")
    trial("fixed top", lambda packed: "top")
    trial("fixed bottom", lambda packed: "bottom")
    trial("read: open edge", open_band)
