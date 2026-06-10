from __future__ import annotations

import random
from dataclasses import dataclass, replace

# Condition variables — all computable from local cell + 6 neighbors (no global view).
VARIABLES = (
    "my_energy",
    "my_age",
    "nutrient_here",
    "toxin_here",
    "live_neighbors",
    "nutrient_neighbor_max",
    "toxin_neighbor_max",
    "free_cells",
)
# Rough domain ceilings per variable, used to bound random thresholds.
VAR_MAX = {
    "my_energy": 100,
    "my_age": 100,
    "nutrient_here": 100,
    "toxin_here": 100,
    "live_neighbors": 6,
    "nutrient_neighbor_max": 100,
    "toxin_neighbor_max": 100,
    "free_cells": 6,
}
OPS = (">", "<")
VERBS = ("metabolize", "divide", "migrate", "dormant")
TARGETS = ("nutrient_max", "toxin_min", "empty_max", "random")
REGULATORS = ("division_threshold", "exploration", "toxin_caution")


@dataclass(frozen=True)
class Condition:
    variable: str
    op: str
    threshold: int


@dataclass(frozen=True)
class Gene:
    cond1: Condition
    cond2: Condition | None
    verb: str
    target: str


@dataclass(frozen=True)
class Genome:
    regulators: dict[str, int]
    behaviors: tuple[Gene, ...]


def _cond_dict(c: Condition | None) -> dict | None:
    if c is None:
        return None
    return {"variable": c.variable, "op": c.op, "threshold": c.threshold}


def to_dict(g: Genome) -> dict:
    return {
        "regulators": g.regulators,
        "behaviors": [
            {
                "cond1": _cond_dict(gene.cond1),
                "cond2": _cond_dict(gene.cond2),
                "verb": gene.verb,
                "target": gene.target,
            }
            for gene in g.behaviors
        ],
    }


def _rand_condition(rng: random.Random) -> Condition:
    var = rng.choice(VARIABLES)
    return Condition(var, rng.choice(OPS), rng.randint(0, VAR_MAX[var]))


def _rand_gene(rng: random.Random) -> Gene:
    cond2 = _rand_condition(rng) if rng.random() < 0.4 else None
    return Gene(_rand_condition(rng), cond2, rng.choice(VERBS), rng.choice(TARGETS))


def random_genome(rng: random.Random, max_genes: int = 8) -> Genome:
    regs = {r: rng.randint(0, 100) for r in REGULATORS}
    n = rng.randint(1, max(1, max_genes // 2))
    return Genome(regs, dedupe(_rand_gene(rng) for _ in range(n)))


def mutate(genome: Genome, rng: random.Random, cfg) -> Genome:
    regs = dict(genome.regulators)
    for r in regs:
        if rng.random() < cfg.p_point:
            delta = rng.randint(1, cfg.point_delta) * rng.choice((-1, 1))
            regs[r] = max(0, min(100, regs[r] + delta))

    behaviors = list(genome.behaviors)
    if behaviors and rng.random() < cfg.p_point:
        i = rng.randrange(len(behaviors))
        behaviors[i] = _point_mutate_gene(behaviors[i], rng, cfg)
    if rng.random() < cfg.p_add and len(behaviors) < cfg.max_genes:
        behaviors.insert(rng.randint(0, len(behaviors)), _rand_gene(rng))
    if rng.random() < cfg.p_del and len(behaviors) > 1:
        behaviors.pop(rng.randrange(len(behaviors)))
    if rng.random() < cfg.p_dup and len(behaviors) < cfg.max_genes:
        # duplicate-and-diverge: a copy that's immediately point-mutated, so it's a
        # new related gene rather than dead-code identical to the original
        behaviors.append(_point_mutate_gene(behaviors[rng.randrange(len(behaviors))], rng, cfg))
    if rng.random() < cfg.p_swap and len(behaviors) > 1:
        i, j = rng.sample(range(len(behaviors)), 2)
        behaviors[i], behaviors[j] = behaviors[j], behaviors[i]

    return Genome(regs, dedupe(behaviors))


def dedupe(behaviors) -> tuple[Gene, ...]:
    """Drop exact-duplicate rules (the CA is first-match-wins, so a repeat is dead
    code wasting a gene slot). Keeps first occurrence and order."""
    seen, out = set(), []
    for g in behaviors:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return tuple(out)


def _tune_threshold(c: Condition, rng: random.Random, cfg) -> Condition:
    delta = rng.randint(1, cfg.point_delta) * rng.choice((-1, 1))
    return replace(c, threshold=max(0, min(VAR_MAX[c.variable], c.threshold + delta)))


def _point_mutate_gene(gene: Gene, rng: random.Random, cfg) -> Gene:
    # Morph a random facet of the gene, not just one threshold — so a gene can
    # gradually change what it senses and what it does, not only fine-tune.
    roll = rng.random()
    if roll < 0.42:  # tune a threshold (cond1 or cond2)
        if gene.cond2 is not None and rng.random() < 0.5:
            return replace(gene, cond2=_tune_threshold(gene.cond2, rng, cfg))
        return replace(gene, cond1=_tune_threshold(gene.cond1, rng, cfg))
    if roll < 0.56:  # flip comparison direction
        return replace(gene, cond1=replace(gene.cond1, op="<" if gene.cond1.op == ">" else ">"))
    if roll < 0.70:  # re-sense: change which variable cond1 reads
        var = rng.choice(VARIABLES)
        return replace(gene, cond1=Condition(var, gene.cond1.op, rng.randint(0, VAR_MAX[var])))
    if roll < 0.83:  # change action
        return replace(gene, verb=rng.choice(VERBS))
    if roll < 0.93:  # change action target
        return replace(gene, target=rng.choice(TARGETS))
    if gene.cond2 is None:  # grow a second condition
        return replace(gene, cond2=_rand_condition(rng))
    return replace(gene, cond2=None)  # drop the second condition


def crossover(g1: Genome, g2: Genome, rng: random.Random, max_genes: int) -> Genome:
    """Single-point recombination of two genomes' rule lists, regulators picked
    per-gene from either parent. Used for horizontal gene transfer on division."""
    b1, b2 = list(g1.behaviors), list(g2.behaviors)
    if b1 and b2:
        behaviors = b1[: rng.randint(0, len(b1))] + b2[rng.randint(0, len(b2)) :]
    else:
        behaviors = b1 or b2
    if not behaviors:
        behaviors = [rng.choice(b1 or b2)]
    regs = {
        r: (g1.regulators[r] if rng.random() < 0.5 else g2.regulators.get(r, g1.regulators[r]))
        for r in g1.regulators
    }
    return Genome(regs, dedupe(behaviors)[:max_genes])
