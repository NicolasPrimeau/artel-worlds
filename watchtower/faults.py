from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from .infra import Infra


# An incident is one concrete firing of a fault family: the family's stable root cause, dressed in
# THIS occurrence's mutated surface — which node, what the dials read, which log strings. The fix is
# a node-resolved action sequence. Two incidents of the same family share `family` and the SHAPE of
# `fix` but differ in `nodes` and numbers, so string-memorizing one does nothing for the next; only
# a runbook that generalizes the root cause carries over. That generalization is what Artel pools.
#
# CRITICAL: every random choice is drawn in spawn() and captured; apply() is PURE. The same spec is
# applied to BOTH fleets' infra, so it must perturb them identically — no rng at apply time.
@dataclass
class IncidentSpec:
    family: str
    title: str
    root: str  # the true root cause, one line — what a good runbook would record
    alert: str  # the pager text: the loudest SYMPTOM, often a degraded dependent, not the cause
    fix: list[tuple[str, str]]  # ordered (action, node) the responder must perform to resolve
    apply: Callable[[Infra], None]  # perturb a fleet's infra into this incident's failed state


@dataclass(frozen=True)
class Fault:
    key: str
    title: str
    root: str
    spawn: Callable[[Random], IncidentSpec]


def _spike(
    key: str,
    title: str,
    root: str,
    nodes: tuple[str, ...],
    metric: str,
    lo: float,
    hi: float,
    action: str,
    logs: Callable[[str, float], list[str]],
    alert: Callable[[str, float], str],
    down: bool = False,
    aux: Callable[[Random], dict] | None = None,
) -> Fault:
    # The common shape: pick one eligible node, drive one metric into the red, the fix is a single
    # remediation on that node. Variety lives in WHICH metric and WHICH remediation — and in the
    # traps, where the loud node is a dependent and the cure is elsewhere. `aux` returns extra
    # preset values (e.g. a deploy bump) so apply() stays pure.
    def spawn(rng: Random) -> IncidentSpec:
        node = rng.choice(nodes)
        val = round(rng.uniform(lo, hi), 1)
        extra = aux(rng) if aux else {}
        loglines = logs(node, val)
        alert_text = alert(node, val)

        def apply(infra: Infra) -> None:
            s = infra.nodes[node]
            s.metrics[metric] = val
            s.status = "down" if down else "degraded"
            s.incident = key
            s.logs = list(loglines)
            for k, v in extra.items():
                if k == "deploy_bump":
                    s.deploy_version += int(v)
                else:
                    s.metrics[k] = v
            infra.propagate()

        return IncidentSpec(key, title, root, alert_text, [(action, node)], apply)

    return Fault(key, title, root, spawn)


def _queue_backlog(rng: Random) -> IncidentSpec:
    # Two steps, order-critical: a consumer wedged and the queue piled up behind it. Drain first and
    # the fresh backlog just re-wedges the same dead consumer — you must restart the CONSUMER, then
    # clear the queue. The alert points at the queue and the api (both loud); the wedged worker is
    # quiet. This is the family where a runbook pays the most: the wrong order looks productive.
    consumer = rng.choice(("worker-1", "worker-2"))
    depth = round(rng.uniform(40_000, 120_000), 0)
    stall = int(rng.uniform(120, 600))
    cpu = round(rng.uniform(1, 5), 1)

    def apply(infra: Infra) -> None:
        w = infra.nodes[consumer]
        w.status, w.incident = "degraded", "queue_backlog"
        w.metrics["cpu_pct"] = cpu  # wedged: pinned at zero throughput
        w.logs = [
            f"consumer stalled: no ack in {stall}s",
            "thread blocked on poisoned message, not advancing offset",
        ]
        q = infra.nodes["queue"]
        q.status, q.incident = "degraded", "queue_backlog"
        q.metrics["queue_depth"] = depth
        q.logs = [f"depth={int(depth)} and climbing", "oldest unacked message age 11m"]
        infra.propagate()

    return IncidentSpec(
        "queue_backlog",
        "Queue backlog behind a wedged consumer",
        "A consumer wedged on a poison message and stopped acking; restart the CONSUMER first, "
        "THEN clear the queue — draining first just re-wedges it.",
        f"PAGE: queue depth {int(depth)} climbing, api p99 1400ms, worker throughput -> 0",
        [("restart", consumer), ("clear_queue", "queue")],
        apply,
    )


def _db_primary_stuck(rng: Random) -> IncidentSpec:
    # Two steps on the data tier: the primary wedged on long locks, connections backing up. Promote
    # the replica to take writes, THEN recycle the stuck primary. Restarting the primary first drops
    # every in-flight write; the failover has to come first. api/auth/workers all scream — red
    # herrings. Mutates the connection magnitude but the root cause and step order are fixed.
    conns = round(rng.uniform(180, 400), 0)
    lat = round(rng.uniform(2000, 6000), 1)

    def apply(infra: Infra) -> None:
        db = infra.nodes["db"]
        db.status, db.incident = "degraded", "db_primary_stuck"
        db.metrics["conns"] = conns
        db.metrics["latency_ms"] = lat
        db.logs = [
            f"{int(conns)} active connections, pool exhausted",
            "longest transaction holding locks for 4m12s",
        ]
        infra.propagate()

    return IncidentSpec(
        "db_primary_stuck",
        "Primary database wedged on long locks",
        "The primary wedged holding locks with connections maxed; failover to the replica to "
        "restore writes, THEN recycle the primary — restarting it first drops in-flight writes.",
        f"PAGE: db connections {int(conns)} maxed, api/auth 5xx surging, writes stalled",
        [("failover", "db"), ("restart", "db")],
        apply,
    )


FAMILIES: tuple[Fault, ...] = (
    _spike(
        "disk_full",
        "Disk saturated on a stateful node",
        "The node's disk filled (logs/old segments); reclaim space to restore writes.",
        ("db", "db-replica", "worker-1", "worker-2"),
        "disk_pct",
        96.0,
        99.9,
        "clear_queue",
        lambda n, v: [f"disk at {v}% — write() returning ENOSPC", "WAL/segment writes failing"],
        lambda n, v: f"PAGE: {n} disk {v}% full, writes failing with ENOSPC",
    ),
    _spike(
        "memory_leak",
        "Memory leak driving GC thrash",
        "Heap leaked until the process is thrashing GC; a restart reclaims it.",
        ("api", "auth", "web", "worker-1", "worker-2"),
        "mem_pct",
        93.0,
        99.0,
        "restart",
        lambda n, v: [f"mem at {v}%, GC pause 800ms+", "OOMKilled risk imminent"],
        lambda n, v: f"PAGE: {n} memory {v}%, latency climbing, GC thrashing",
    ),
    _spike(
        "cache_down",
        "Cache process crashed",
        "The cache process died; every read falls through to the db until it's restarted.",
        ("cache",),
        "error_rate",
        80.0,
        100.0,
        "restart",
        lambda n, v: ["connection refused on :6379", "process not responding to PING"],
        lambda n, v: "PAGE: cache unreachable, db CPU spiking from read fall-through",
        down=True,
    ),
    _spike(
        "bad_deploy",
        "Regression in the latest deploy",
        "The last deploy shipped a regression; error rate jumped on version bump — roll it back.",
        ("api", "web", "auth"),
        "error_rate",
        15.0,
        60.0,
        "rollback",
        lambda n, v: [f"5xx rate {v}% since v-bump", "NullPointer in new request handler path"],
        lambda n, v: f"PAGE: {n} error rate {v}% spiked right after a deploy",
        aux=lambda rng: {"deploy_bump": 1, "latency_ms": round(rng.uniform(300, 900), 1)},
    ),
    _spike(
        "conn_pool_exhausted",
        "Connection pool leaked on the caller",
        "The CALLER leaked its db connection pool; restart the caller to drain it — the db is fine.",
        ("api", "auth"),
        "conns",
        95.0,
        100.0,
        "restart",
        lambda n, v: [
            f"pool {v}% checked-out, 0 idle",
            "waiters timing out acquiring a connection",
        ],
        lambda n, v: f"PAGE: {n} cannot get db connections, pool {v}% exhausted",
        aux=lambda rng: {"error_rate": round(rng.uniform(8, 22), 1)},
    ),
    _spike(
        "replica_lag",
        "Read replica fell behind",
        "The replica lagged far behind the primary, serving stale reads; failover/resync it.",
        ("db-replica",),
        "replica_lag_s",
        45.0,
        400.0,
        "failover",
        lambda n, v: [f"replication lag {v}s", f"serving reads {v}s stale"],
        lambda n, v: f"PAGE: db-replica lag {v}s, users seeing stale data",
    ),
    _spike(
        "cpu_saturation",
        "CPU saturated under sustained load",
        "Sustained load pinned CPU and throughput collapsed; scale out — a restart just refills.",
        ("worker-1", "worker-2", "api"),
        "cpu_pct",
        97.0,
        100.0,
        "scale",
        lambda n, v: [f"CPU {v}%, run-queue backing up", "p99 latency 3x baseline under load"],
        lambda n, v: f"PAGE: {n} CPU {v}% pinned, throughput collapsing",
    ),
    _spike(
        "cert_expiry",
        "TLS certificate expired at the edge",
        "An expired TLS cert is failing handshakes at the edge; rotate the certificate.",
        ("lb", "auth"),
        "error_rate",
        70.0,
        100.0,
        "rotate",
        lambda n, v: ["TLS handshake failure: certificate expired", "clients rejecting connection"],
        lambda n, v: f"PAGE: {n} TLS handshakes failing, certificate expired",
        down=True,
    ),
    _spike(
        "dependency_crash",
        "Backing queue service crashed",
        "The queue process crashed; workers and api scream but are fine — restart the DEPENDENCY.",
        ("queue",),
        "error_rate",
        80.0,
        100.0,
        "restart",
        lambda n, v: ["broker process exited, port closed", "producers failing to enqueue"],
        lambda n, v: "PAGE: workers idle + api 5xx — queue broker appears down",
        down=True,
    ),
    _spike(
        "traffic_spike",
        "Organic traffic surge at the front tier",
        "A real load surge overwhelmed the front tier; scale it out — nothing is broken, just small.",
        ("web", "lb"),
        "latency_ms",
        900.0,
        3000.0,
        "scale",
        lambda n, v: [f"p99 {v}ms, 2.4x normal RPS", "connection queue saturating"],
        lambda n, v: f"PAGE: {n} p99 {v}ms under a traffic surge",
    ),
    Fault(
        "queue_backlog",
        "Queue backlog behind a wedged consumer",
        "A consumer wedged on a poison message; restart the consumer, then clear the queue.",
        _queue_backlog,
    ),
    Fault(
        "db_primary_stuck",
        "Primary database wedged on long locks",
        "The primary wedged on locks; failover to the replica, then recycle the primary.",
        _db_primary_stuck,
    ),
)


def family_keys() -> list[str]:
    return [f.key for f in FAMILIES]
