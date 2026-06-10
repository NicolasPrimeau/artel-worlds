"""Measure evolutionary pressure: run headless (no reset) and watch whether the
genome distribution shifts away from random toward survival-favoring strategies."""

from automata.agent import HeuristicAgent
from automata.config import DEFAULT
from automata.tick import step
from automata.world import World


def toxin_aware(g):
    # genome can sense toxin AND act to escape it
    senses = any(
        c and c.variable in ("toxin_here", "toxin_neighbor_max")
        for gene in g.behaviors
        for c in (gene.cond1, gene.cond2)
    )
    escapes = any(gene.verb == "migrate" and gene.target == "toxin_min" for gene in g.behaviors)
    return senses and escapes


def forager(g):
    return any(
        gene.verb in ("migrate", "divide") and gene.target == "nutrient_max" for gene in g.behaviors
    )


def verb_mix(orgs):
    from collections import Counter

    c = Counter(gene.verb for o in orgs for gene in o.genome.behaviors)
    tot = sum(c.values()) or 1
    return {v: round(c[v] / tot, 2) for v in ("metabolize", "divide", "migrate", "dormant")}


def snapshot(world, label):
    orgs = list(world.organisms.values())
    if not orgs:
        print(f"{label:>10}: EXTINCT")
        return
    pop = len(orgs)
    lineages = len({o.lineage_id for o in orgs})
    uniq = len({(tuple(sorted(o.genome.regulators.items())), o.genome.behaviors) for o in orgs})
    tox = sum(toxin_aware(o.genome) for o in orgs) / pop
    forg = sum(forager(o.genome) for o in orgs) / pop
    mean_age = sum(o.age for o in orgs) / pop
    mean_len = sum(len(o.genome.behaviors) for o in orgs) / pop
    dom = (
        max(
            (sum(1 for o in orgs if o.lineage_id == lid) for lid in {o.lineage_id for o in orgs}),
            default=0,
        )
        / pop
    )
    print(
        f"{label:>10}: pop={pop:>4} lin={lineages:>3} uniqG={uniq:>4} "
        f"toxin_aware={tox:>4.0%} forager={forg:>4.0%} meanAge={mean_age:>5.1f} "
        f"meanGenes={mean_len:>3.1f} domLin={dom:>4.0%} {verb_mix(orgs)}"
    )


w = World(DEFAULT, seed=7)
w.seed(DEFAULT.initial_population)
agent = HeuristicAgent()
snapshot(w, "tick 0")
for target in (100, 300, 600, 1000, 1500):
    while w.tick_count < target:
        step(w, agent)
    snapshot(w, f"tick {target}")
