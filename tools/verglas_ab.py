from __future__ import annotations

# The meeting is adversarial: the impostor lies and the crew doubt. Three regimes, same board:
#   siloed        — no Artel conduit; each agent votes on its own slice.
#   pooled (blind)— testimony pooled and trusted at face value (the most-accused goes out).
#   pooled (doubt)— testimony pooled, BUT the impostor counter-accuses and crew apply reasonable
#                   doubt, so lone hearsay can't convict and lies sometimes land.
# Blind trust overstates the crew; doubt is honest and shows the ceiling of a FIXED rule — the gap
# between doubt and blind is the room the LLM has to earn by actually reasoning about who's lying.

from collections import Counter

from verglas.brain import make_decider
from verglas.engine import play


def trial(label, share, doubt, n=400, impostors=1, agents=6):
    decide = make_decider(share, doubt)
    crew = imp = 0
    by = Counter()
    wrong = meet = 0
    for s in range(n):
        g = play(s, decide, n=agents, impostors=impostors)
        crew += g.winner == "crew"
        imp += g.winner == "impostor"
        by[(g.winner, g.win_by)] += 1
        wrong += g.wrong_ejections
        meet += len(g.meetings)
    print(
        f"{label:<22} crew {round(crew / n * 100):>3}% | imp {round(imp / n * 100):>3}% | "
        f"eject-win {round(by[('crew', 'ejection')] / n * 100):>3}% | "
        f"wrong-eject/g {wrong / n:.2f} | meetings/g {meet / n:.1f}"
    )


def suite(agents, impostors):
    print(f"=== {agents} agents, {impostors} impostor(s) ===")
    trial("siloed (no Artel)", False, True, agents=agents, impostors=impostors)
    trial("pooled (blind trust)", True, False, agents=agents, impostors=impostors)
    trial("pooled (doubt+lies)", True, True, agents=agents, impostors=impostors)


if __name__ == "__main__":
    suite(6, 1)
    suite(8, 2)
