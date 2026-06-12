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


def random_genome(rng: random.Random, max_genes: int = 8) -> Genome:
    # A full strand: one gene per sensory dimension (VARIABLE), each with a randomized comparison
    # and action. Every cell therefore starts with COMPLETE DNA — and a tribe is seeded from a
    # single shared roll of this strand (see World.seed), so its founders are genetically identical
    # until mutation and selection pull them apart. The dimension set fixes the length; max_genes
    # is accepted for signature compatibility but the strand is always one gene per VARIABLE.
    regs = {r: rng.randint(0, 100) for r in REGULATORS}
    genes = tuple(
        Gene(
            Condition(var, rng.choice(OPS), rng.randint(0, VAR_MAX[var])),
            None,
            rng.choice(VERBS),
            rng.choice(TARGETS),
        )
        for var in VARIABLES
    )
    return Genome(regs, genes)


def mutate(genome: Genome, rng: random.Random, cfg) -> Genome:
    # Dimension-preserving: the strand always keeps exactly one gene per VARIABLE, so a cell can
    # never lose DNA. Mutation tunes regulators and rewrites genes IN PLACE (threshold, comparison,
    # verb, target); it never adds, drops, or re-senses a gene — that would break the
    # one-gene-per-dimension invariant and could leave a lineage with empty DNA over generations.
    regs = dict(genome.regulators)
    for r in regs:
        if rng.random() < cfg.p_point:
            delta = rng.randint(1, cfg.point_delta) * rng.choice((-1, 1))
            regs[r] = max(0, min(100, regs[r] + delta))
    genes = tuple(
        _point_mutate_gene(g, rng, cfg) if rng.random() < cfg.p_point else g
        for g in genome.behaviors
    )
    return Genome(regs, genes)


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
    # Rewrite one facet of the gene while KEEPING the dimension it senses (cond1.variable) fixed —
    # tune the threshold, flip the comparison, or change the action / its target. The variable is
    # never reassigned, so the gene stays bound to its dimension for the life of the strand.
    roll = rng.random()
    if roll < 0.45:  # tune the threshold on this gene's dimension
        return replace(gene, cond1=_tune_threshold(gene.cond1, rng, cfg))
    if roll < 0.62:  # flip the comparison direction
        return replace(gene, cond1=replace(gene.cond1, op="<" if gene.cond1.op == ">" else ">"))
    if roll < 0.81:  # change the action this dimension triggers
        return replace(gene, verb=rng.choice(VERBS))
    return replace(gene, target=rng.choice(TARGETS))  # change the action's target


def crossover(g1: Genome, g2: Genome, rng: random.Random, max_genes: int) -> Genome:
    """Per-dimension recombination: both parents are dimension-aligned strands (one gene per
    VARIABLE, same order), so each gene of the child is inherited from one parent or the other at
    the SAME dimension. Regulators are picked per key. The child keeps the full strand."""
    n = min(len(g1.behaviors), len(g2.behaviors))
    genes = tuple(g1.behaviors[i] if rng.random() < 0.5 else g2.behaviors[i] for i in range(n))
    if len(g1.behaviors) > n:  # parents should be equal-length; keep any extra g1 dimensions
        genes = genes + tuple(g1.behaviors[n:])
    regs = {
        r: (g1.regulators[r] if rng.random() < 0.5 else g2.regulators.get(r, g1.regulators[r]))
        for r in g1.regulators
    }
    return Genome(regs, genes)
