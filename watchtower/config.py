from __future__ import annotations

from dataclasses import dataclass


# A small service graph (10 nodes) with a real dependency spine: traffic enters at the load
# balancer and fans down through web -> api -> {db, cache, queue} -> workers. A node is only as
# healthy as the dependencies it leans on, so one sick node ripples UP into degraded callers —
# which is exactly what makes diagnosis non-trivial: the loudest alarm is rarely the root cause.
@dataclass(frozen=True)
class Node:
    name: str
    kind: str  # lb | web | api | db | cache | queue | worker | auth
    depends_on: tuple[str, ...] = ()


NODES: tuple[Node, ...] = (
    Node("lb", "lb", ("web",)),
    Node("web", "web", ("api",)),
    Node("api", "api", ("db", "cache", "queue", "auth")),
    Node("auth", "auth", ("db",)),
    Node("db", "db", ()),
    Node("db-replica", "db", ("db",)),
    Node("cache", "cache", ()),
    Node("queue", "queue", ()),
    Node("worker-1", "worker", ("queue", "db")),
    Node("worker-2", "worker", ("queue", "db")),
)


# Action repertoire shared by both fleets. Diagnostics are cheap and never change the world;
# remediations cost real time and only the RIGHT one on the RIGHT node advances an incident.
# Times are in simulated seconds — MTTR is the sum of them, so wasted steps are the whole cost.
DIAGNOSTIC_ACTIONS = ("inspect", "read_logs")
REMEDIATION_ACTIONS = ("restart", "scale", "rollback", "clear_queue", "failover", "rotate")
ACTIONS = DIAGNOSTIC_ACTIONS + REMEDIATION_ACTIONS

ACTION_SECONDS: dict[str, float] = {
    "inspect": 8.0,
    "read_logs": 8.0,
    "restart": 30.0,
    "scale": 45.0,
    "rollback": 60.0,
    "clear_queue": 25.0,
    "failover": 50.0,
    "rotate": 40.0,
}

# A remediation aimed at the wrong node or of the wrong kind doesn't fix anything and burns this
# much extra on top of its base cost (rollback that didn't help, a needless restart). Blind
# flailing is how a solo responder's MTTR balloons; a recalled runbook skips straight past it.
WRONG_ACTION_PENALTY = 15.0

# Every incident carries a fixed detection lead before a responder can act, plus a hard ceiling
# on how many actions one responder may take — past it the incident is logged as a miss at a
# punishing MTTR so a flailing responder can't stall the stream or hide a failure.
DETECTION_SECONDS = 10.0
MAX_ACTIONS_PER_INCIDENT = 14
UNRESOLVED_MTTR = 900.0


@dataclass(frozen=True)
class Config:
    nodes: tuple[Node, ...] = NODES
    fleet_size: int = 3  # responders per fleet; incidents round-robin across them, both sides alike
    # Seconds between incidents firing into BOTH fleets at once. Always-on and event-driven, so the
    # weeks-long curve accrues cheaply; the per-day spend cap is the real budget guard.
    incident_interval: float = 1500.0
    incident_jitter: float = 300.0

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)

    @property
    def node_by_name(self) -> dict[str, Node]:
        return {n.name: n for n in self.nodes}


DEFAULT = Config()
