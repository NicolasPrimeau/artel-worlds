from __future__ import annotations

# Does pooling what agents saw (the Artel conduit) actually win the game? Crew win-rate with testimony
# SHARED vs each agent siloed, over many seeds, on the faithful board (tasks can also win it). This is
# the load-bearing test: communication is the crew's edge in the meeting, so cutting it should hand
# games to the impostor — and we split crew wins by HOW they won to see comms drive ejection wins.

from collections import Counter

from alibi.brain import make_decider
from alibi.engine import play


def trial(label, share, n=400, impostors=1, agents=6):
    decide = make_decider(share)
    crew = imp = 0
    by = Counter()
    ej = wrong = meet = 0
    for s in range(n):
        g = play(s, decide, n=agents, impostors=impostors)
        crew += g.winner == "crew"
        imp += g.winner == "impostor"
        by[(g.winner, g.win_by)] += 1
        ej += g.ejected_impostors
        wrong += g.wrong_ejections
        meet += len(g.meetings)
    print(
        f"{label:<26} crew {round(crew / n * 100)}% | imp {round(imp / n * 100)}% | "
        f"eject-win {round(by[('crew', 'ejection')] / n * 100)}% task-win "
        f"{round(by[('crew', 'tasks')] / n * 100)}% | wrong-eject/g {wrong / n:.2f} | "
        f"meetings/g {meet / n:.1f}"
    )


if __name__ == "__main__":
    print("=== 6 agents, 1 impostor ===")
    trial("Artel ON  (pool)", True)
    trial("Artel OFF (siloed)", False)
    print("=== 8 agents, 2 impostors ===")
    trial("Artel ON  (pool)", True, agents=8, impostors=2)
    trial("Artel OFF (siloed)", False, agents=8, impostors=2)
