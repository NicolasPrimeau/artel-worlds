from __future__ import annotations

from random import Random

from .config import (
    ACTION_SECONDS,
    DETECTION_SECONDS,
    DIAGNOSTIC_ACTIONS,
    MAX_ACTIONS_PER_INCIDENT,
    REMEDIATION_ACTIONS,
    UNRESOLVED_MTTR,
    WRONG_ACTION_PENALTY,
)
from .faults import FAMILIES, IncidentSpec
from .infra import Infra


EPOCH_INCIDENTS = 40


def spec_for(seed: int, seq: int) -> IncidentSpec:
    # Each incident is its OWN seeded draw on (seed, seq) — deterministic, O(1), and identical for
    # both fleets. No fast-forward after a restart: the world just resumes at the next seq and the
    # same incident comes up that always would have. The cursor is simply how many have been done.
    # The epoch is the world's evolution clock: every EPOCH_INCIDENTS, new roots open and priors
    # drift — the environment stays ahead of any single responder's exposure rate, so the frontier
    # is mappable by a fleet that pools exploration but never by one agent alone.
    rng = Random(f"{seed}:{seq}")
    return rng.choice(FAMILIES).spawn(rng, seq // EPOCH_INCIDENTS)


def make_stream(seed: int, length: int) -> list[IncidentSpec]:
    # The PAIRED stream as a list, for tests and offline replay — both fleets face byte-identical
    # incidents in the same order, reproducible across restarts so the weeks-long curve is one world.
    return [spec_for(seed, i) for i in range(length)]


def storm_for(seed: int, seq: int, k: int) -> list[IncidentSpec]:
    # A STORM is up to k node-disjoint incidents drawn from k consecutive seqs — deterministic and
    # resumable exactly like the single stream, just consumed k-at-a-time. Disjoint fix targets so
    # each incident owns its node and resolves independently on the one shared infra a fleet holds.
    out: list[IncidentSpec] = []
    taken: set[str] = set()
    for i in range(max(1, k)):
        spec = spec_for(seed, seq + i)
        nodes = {n for _, n in spec.fix}
        if nodes & taken:
            continue
        taken |= nodes
        out.append(spec)
    return out or [spec_for(seed, seq)]


class Incident:
    """One fleet's live handling of one incident. Holds that fleet's infra (already perturbed into
    the failed state), tracks the responder's actions, and accrues the simulated clock. MTTR is the
    sum of action times: a responder who recalls the runbook spends it on the fix; one who doesn't
    spends it inspecting the wrong (loud) node and trying remediations that don't take."""

    def __init__(self, spec: IncidentSpec, seq: int, infra: Infra, fleet: str):
        self.spec = spec
        self.seq = seq
        self.infra = infra
        self.fleet = fleet
        self.actions: list[dict] = []
        self.elapsed = 0.0
        self.step_i = 0
        self.resolved = False
        self.missed = False
        # real ops time is noisy: every duration draws from a normal around its base cost,
        # clamped to [0.5x, 2x]. Seeded per (seq, fleet) so replays and tests reproduce; both
        # fleets draw from identical distributions, so the noise is unbiased between arms.
        self._rng = Random(f"time:{seq}:{fleet}:{spec.family}")
        self._detection = self._cost(DETECTION_SECONDS)
        spec.apply(infra)

    def _cost(self, base: float) -> float:
        return round(min(base * 2.0, max(base * 0.5, self._rng.gauss(base, base * 0.25))), 1)

    @property
    def family(self) -> str:
        return self.spec.family

    def _heal_all(self) -> None:
        for name, s in list(self.infra.nodes.items()):
            if s.incident == self.family:
                self.infra.heal(name)
        self.infra.propagate()

    def act(self, action: str, node: str) -> dict:
        if self.resolved or self.missed:
            return {"error": "incident already closed"}
        action = (action or "").strip()
        node = (node or "").strip()
        if action in DIAGNOSTIC_ACTIONS:
            self.elapsed += self._cost(ACTION_SECONDS[action])
            out = self.infra.inspect(node) if action == "inspect" else self.infra.read_logs(node)
            self._record(action, node, "ok")
            return out
        if action not in REMEDIATION_ACTIONS:
            self.elapsed += self._cost(5.0)
            self._record(action, node, "unknown action")
            return {"error": f"unknown action '{action}'", "valid": list(ACTION_SECONDS)}
        if node not in self.infra.nodes:
            self.elapsed += self._cost(5.0)
            self._record(action, node, "no such node")
            return {"error": f"no node named '{node}'"}
        want_action, want_node = self.spec.fix[self.step_i]
        if (
            action == want_action == "failover"
            and want_node == "db"
            and node in ("db", "db-replica")
        ):
            # promoting the replica and failing over the primary are the same operation;
            # punishing the synonym made db_primary_stuck an unlearnable black hole
            node = want_node
        if action == want_action and node == want_node:
            self.elapsed += self._cost(ACTION_SECONDS[action])
            self.step_i += 1
            self._record(action, node, "applied")
            if self.step_i >= len(self.spec.fix):
                self._heal_all()
                self.resolved = True
                return {
                    "result": f"{action} on {node} applied — incident RESOLVED",
                    "resolved": True,
                }
            return {"result": f"{action} on {node} applied — partial, more remains"}
        self.elapsed += self._cost(ACTION_SECONDS[action]) + self._cost(WRONG_ACTION_PENALTY)
        self._record(action, node, "no effect")
        return {"result": f"{action} on {node} had no effect — symptoms persist"}

    def _record(self, action: str, node: str, result: str) -> None:
        self.actions.append({"action": action, "node": node, "result": result})
        if len(self.actions) >= MAX_ACTIONS_PER_INCIDENT and not self.resolved:
            # a responder that can't get there is cut off so it can neither stall the stream nor
            # bury the failure: auto-remediate the world and book it as a miss at the ceiling MTTR
            self._heal_all()
            self.missed = True

    def finalize(self) -> None:
        # the responder stopped (round budget spent, gave up, or errored) without resolving: heal the
        # world so the next incident starts clean and book this one honestly as a miss at the ceiling
        if not self.resolved and not self.missed:
            self._heal_all()
            self.missed = True

    def mttr(self) -> float:
        if self.missed:
            return UNRESOLVED_MTTR
        return self._detection + self.elapsed

    def view(self) -> dict:
        return {
            "seq": self.seq,
            "family": self.family,
            "title": self.spec.title,
            "fleet": self.fleet,
            "alert": self.spec.alert,
            "actions": list(self.actions),
            "mttr": round(self.mttr(), 1),
            "resolved": self.resolved,
            "missed": self.missed,
        }
