from __future__ import annotations

# A/B: does the Artel coordination edge beat baseline? HYBRID model — the deterministic game-management
# reflex (commit / low-block off the live score) runs EVERY TICK and is identical for any coached side;
# only the strategic flank read differs (heuristic here, the real LLM in ab_llm.py) and it refreshes on
# the same slow cadence for both, so the comparison is apples-to-apples (no side gets faster reactivity).
# Sides alternate per seed to cancel home-field bias.

import sys

from pitch import bot, commander
from pitch.engine import Pitch

CADENCE = (
    75  # ticks (~6s) between flank re-reads — the brain's slow loop, same for LLM and heuristic
)


def hybrid_brain(team, flank_fn):
    st = {"tick": -1, "oy": None}

    def brain(pitch, p):
        if pitch.tick != st["tick"]:
            st["tick"] = pitch.tick
            if st["oy"] is None or pitch.tick % CADENCE == 0:
                st["oy"] = flank_fn(pitch, team)  # strategic read, on the slow cadence
        if p.team == team:
            commit, low_block = commander.game_management(pitch, team)  # reflex, EVERY tick
            plan = commander.Plan(st["oy"], commit, low_block, False)
            return commander.coordinated_decide(pitch, p, plan)
        return bot.decide(pitch, p)

    return brain


def heuristic_flank(pitch, team):
    return commander.plan_for(pitch, team).overload_y


def run(label, brain_for, seeds=70):
    w = ll = d = gf = ga = n = 0
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
            w += cf > ca
            ll += ca > cf
            d += cf == ca
        print(f"\r{label}: {s + 1}/{seeds}", end="", file=sys.stderr)
    wr = round(w / (w + ll) * 100) if (w + ll) else 0
    print(f"\r{label:<36} {w}W {ll}L {d}D | win {wr}% | goals {gf / n:.2f} vs {ga / n:.2f} (n={n})")


if __name__ == "__main__":
    run("baseline vs baseline (control)", lambda t: lambda pi, p: bot.decide(pi, p))
    run("hybrid: heuristic flank + reflex", lambda t: hybrid_brain(t, heuristic_flank))
