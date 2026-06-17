from __future__ import annotations

# Does pooling what agents saw (the Artel job) actually win the game? Crew win-rate with testimony
# SHARED vs each agent siloed, over many seeds. This is the load-bearing test for the social-deduction
# world: communication is the crew's only weapon, so cutting it should hand the game to the mole.

from mole.engine import play


def trial(label, share, n=400, moles=1, agents=6):
    crew = mole = 0
    ej = wrong = meet = 0
    for s in range(n):
        g = play(s, share, n=agents, moles=moles)
        crew += g.winner == "crew"
        mole += g.winner == "mole"
        ej += g.ejected_moles
        wrong += g.wrong_ejections
        meet += g.meetings
    print(
        f"{label:<28} crew {round(crew / n * 100)}% | mole {round(mole / n * 100)}% | "
        f"moles caught/g {ej / n:.2f} | wrong ejects/g {wrong / n:.2f} | meetings/g {meet / n:.1f}"
    )


if __name__ == "__main__":
    print("=== 6 agents, 1 mole ===")
    trial("Artel ON  (pool testimony)", True)
    trial("Artel OFF (each siloed)", False)
    print("=== 8 agents, 2 moles ===")
    trial("Artel ON  (pool testimony)", True, agents=8, moles=2)
    trial("Artel OFF (each siloed)", False, agents=8, moles=2)
