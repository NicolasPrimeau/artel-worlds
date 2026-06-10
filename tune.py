"""Dev tuning harness: sweep params, report trajectory shape to find the
oscillating edge-of-chaos regime (not extinction, not saturation)."""

import dataclasses
import statistics

from worlds.config import DEFAULT
from worlds.tick import step
from worlds.world import World

CAP = DEFAULT.width * DEFAULT.height
TICKS = 300

CANDIDATES = {
    "toxic_waves": dict(
        toxin_emission=6, toxin_degradation=1, toxin_lethal=50, nutrient_regrowth=1
    ),
    "lean_growth": dict(
        gain_per_nutrient=1,
        nutrient_regrowth=1,
        toxin_emission=6,
        toxin_degradation=2,
        toxin_lethal=55,
    ),
    "lean_toxic": dict(
        gain_per_nutrient=1,
        nutrient_regrowth=1,
        toxin_emission=7,
        toxin_degradation=1,
        toxin_lethal=45,
    ),
    "expensive_div": dict(
        cost_division=20,
        gain_per_nutrient=1,
        nutrient_regrowth=1,
        toxin_emission=6,
        toxin_degradation=2,
        toxin_lethal=50,
    ),
    "lean_consume": dict(
        gain_per_nutrient=1,
        consumption_max=5,
        nutrient_regrowth=1,
        toxin_emission=6,
        toxin_degradation=1,
        toxin_lethal=50,
    ),
}

_BLK = " ▁▂▃▄▅▆▇█"


def spark(pops, n=40):
    step_i = max(1, len(pops) // n)
    sample = pops[::step_i]
    hi = max(sample) or 1
    return "".join(_BLK[min(8, v * 8 // hi)] for v in sample)


def trajectory(overrides):
    cfg = dataclasses.replace(DEFAULT, **overrides)
    w = World(cfg, seed=1)
    w.seed(cfg.initial_population)
    pops = []
    for _ in range(TICKS):
        s = step(w)
        pops.append(s["population"])
        if s["population"] == 0:
            break
    return pops


def summarize(name, pops):
    if pops[-1] == 0:
        print(f"{name:>16}: EXTINCT at tick {len(pops)}")
        return
    mx, mn, final = max(pops), min(pops[10:] or pops), pops[-1]
    sat = mx / CAP
    # direction changes in the second half = oscillation proxy
    tail = pops[len(pops) // 2 :]
    turns = sum(
        1 for i in range(1, len(tail) - 1) if (tail[i] - tail[i - 1]) * (tail[i + 1] - tail[i]) < 0
    )
    sd = round(statistics.pstdev(tail))
    print(
        f"{name:>14}: final={final:>4} min={mn:>4} max={mx:>4} sat={sat:>3.0%} "
        f"sd={sd:>4} turns={turns:>3}  {spark(pops)}"
    )


for name, ov in CANDIDATES.items():
    summarize(name, trajectory(ov))
