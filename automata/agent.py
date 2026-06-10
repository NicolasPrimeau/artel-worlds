from __future__ import annotations

from typing import Protocol

from .genome import Condition, Genome

# An organism's decision. The world resolves legality + physics.
Intention = tuple[str, str]  # (verb, target)


class Agent(Protocol):
    """The decision-maker contract. A cellular-automaton brain (HeuristicAgent)
    and a real LLM agent are interchangeable: both turn a perception + genome
    into an Intention. The LLM version does it via a model call (and, when remote,
    reaches the world through the same perceive()/submit() HTTP surface)."""

    def act(self, perception: dict[str, int], genome: Genome) -> Intention: ...


def _eval(cond: Condition, p: dict[str, int]) -> bool:
    v = p[cond.variable]
    return v > cond.threshold if cond.op == ">" else v < cond.threshold


class HeuristicAgent:
    """The cellular automaton. Walks the genome's ordered rules and returns the
    first whose condition holds — the hand-coded stand-in for the LLM, used for
    fast tuning. Same Agent contract the LLM implementation will satisfy."""

    def act(self, perception: dict[str, int], genome: Genome) -> Intention:
        for gene in genome.behaviors:
            if _eval(gene.cond1, perception) and (
                gene.cond2 is None or _eval(gene.cond2, perception)
            ):
                return (gene.verb, gene.target)
        return ("metabolize", "random")
