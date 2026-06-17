from __future__ import annotations

# A/B: does the Artel coordination edge actually beat the baseline? Same deterministic motor both
# sides; the coordinated side runs the tactical edge (overload the weak flank / commit when chasing)
# and optionally the play actuator (LLM-called give-and-gos). Sides alternate to cancel any bias.

import sys

from pitch import bot, commander, plays
from pitch.engine import Pitch


def coord_brain(team, use_plays):
    mgr = plays.PlayManager(team)
    st = {"tick": -1, "plan": None}

    def brain(pitch, p):
        if pitch.tick != st["tick"]:
            st["tick"] = pitch.tick
            plan = commander.plan_for(pitch, team)
            plan.combos = use_plays
            st["plan"] = plan
            if use_plays:
                side = "left" if plan.overload_y < pitch.cfg.width / 2 else "right"
                mgr.call = {"combos": True, "channel": side}
                mgr.update(pitch)
        if p.team == team:
            if use_plays:
                it = mgr.intent(pitch, p)
                if it is not None:
                    return it
            return commander.coordinated_decide(pitch, p, st["plan"])
        return bot.decide(pitch, p)

    return brain


def run(label, brain_for, seeds=70):
    # PAIRED: the coordinated side plays home AND away on each seed, so any home-field bias cancels.
    w = ll = d = 0
    gf = ga = 0
    n = 0
    for s in range(seeds):
        for coord in ("home", "away"):
            opp = "away" if coord == "home" else "home"
            p = Pitch(seed=1000 + s)
            p.setup(["x"] * 9, ["y"] * 9)
            brain = brain_for(coord)
            while p.tick < p.cfg.match_ticks:
                p.step(brain)
            cf, ca = p.score[coord], p.score[opp]
            gf += cf
            ga += ca
            n += 1
            if cf > ca:
                w += 1
            elif ca > cf:
                ll += 1
            else:
                d += 1
        print(f"\r{label}: {s + 1}/{seeds}", end="", file=sys.stderr)
    wr = round(w / (w + ll) * 100) if (w + ll) else 0
    print(f"\r{label:<34} {w}W {ll}L {d}D | win {wr}% | goals {gf / n:.2f} vs {ga / n:.2f} (n={n})")


if __name__ == "__main__":
    run("baseline vs baseline (control)", lambda t: lambda pi, p: bot.decide(pi, p))
    run("positional edge only", lambda t: coord_brain(t, use_plays=False))
    run("positional + LLM-called plays", lambda t: coord_brain(t, use_plays=True))
