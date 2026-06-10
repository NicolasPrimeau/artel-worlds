from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

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


def _rand_condition(rng: random.Random) -> Condition:
    var = rng.choice(VARIABLES)
    return Condition(var, rng.choice(OPS), rng.randint(0, VAR_MAX[var]))


def _rand_gene(rng: random.Random) -> Gene:
    cond2 = _rand_condition(rng) if rng.random() < 0.4 else None
    return Gene(_rand_condition(rng), cond2, rng.choice(VERBS), rng.choice(TARGETS))


def random_genome(rng: random.Random, max_genes: int = 8) -> Genome:
    regs = {r: rng.randint(0, 100) for r in REGULATORS}
    n = rng.randint(1, max(1, max_genes // 2))
    return Genome(regs, tuple(_rand_gene(rng) for _ in range(n)))


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
        behaviors.append(behaviors[rng.randrange(len(behaviors))])
    if rng.random() < cfg.p_swap and len(behaviors) > 1:
        i, j = rng.sample(range(len(behaviors)), 2)
        behaviors[i], behaviors[j] = behaviors[j], behaviors[i]

    return Genome(regs, tuple(behaviors))


def _point_mutate_gene(gene: Gene, rng: random.Random, cfg) -> Gene:
    c = gene.cond1
    delta = rng.randint(1, cfg.point_delta) * rng.choice((-1, 1))
    new_thr = max(0, min(VAR_MAX[c.variable], c.threshold + delta))
    return replace(gene, cond1=replace(c, threshold=new_thr))
