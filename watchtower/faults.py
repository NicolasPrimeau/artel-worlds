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


def _deploy_regression(rng: Random) -> IncidentSpec:
    # One PAGE, two possible truths. Error rate jumps minutes after a deploy on a service node.
    # Usually it IS the deploy (roll back). But sometimes the deploy is innocent: its restart rush
    # surfaced a connection-pool leak, and a rollback changes nothing — the pool needs draining.
    # The alert is identical either way; only read_logs/inspect on the node tells them apart. A good
    # runbook records the DISCRIMINATOR, not a reflex.
    node = rng.choice(("api", "web", "auth"))
    err = round(rng.uniform(15.0, 60.0), 1)
    lat = round(rng.uniform(300, 900), 1)
    pool = round(rng.uniform(95.0, 100.0), 1)
    is_regression = rng.random() < 0.6

    def apply(infra: Infra) -> None:
        s = infra.nodes[node]
        s.status, s.incident = "degraded", "deploy_regression"
        s.metrics["error_rate"] = err
        s.metrics["latency_ms"] = lat
        s.deploy_version += 1
        if is_regression:
            s.logs = [
                f"5xx rate {err}% since version bump",
                f"NullPointer in NEW request handler path (v{s.deploy_version})",
            ]
        else:
            s.metrics["conns"] = pool
            s.logs = [
                f"5xx rate {err}% since restart storm",
                f"pool {pool}% checked-out, 0 idle — waiters timing out acquiring a connection",
                "no errors attributable to new code paths",
            ]
        infra.propagate()

    if is_regression:
        root = "The deploy shipped a regression (new-code stack traces in the logs); roll it back."
        fix = [("rollback", node)]
    else:
        root = (
            "The deploy was innocent: its restart rush surfaced a leaked connection pool — "
            "rollback does nothing; restart the node to drain the pool."
        )
        fix = [("restart", node)]
    return IncidentSpec(
        "deploy_regression",
        "Errors spiking after a deploy",
        root,
        f"PAGE: {node} error rate {err}% spiked minutes after a deploy",
        fix,
        apply,
    )


def _stale_reads(rng: Random) -> IncidentSpec:
    # Users see stale data. The page never names a node — the root is either the read replica
    # lagging (failover) or the cache serving expired keys because invalidation wedged (restart
    # cache). Checking db-replica's replication lag settles it in one inspect.
    is_lag = rng.random() < 0.55
    lag = round(rng.uniform(45.0, 400.0), 1)

    def apply(infra: Infra) -> None:
        if is_lag:
            r = infra.nodes["db-replica"]
            r.status, r.incident = "degraded", "stale_reads"
            r.metrics["replica_lag_s"] = lag
            r.logs = [f"replication lag {lag}s and climbing", f"serving reads {lag}s stale"]
        else:
            c = infra.nodes["cache"]
            c.status, c.incident = "degraded", "stale_reads"
            c.metrics["stale_keys_pct"] = round(40 + lag / 10, 1)
            c.logs = [
                "invalidation queue stalled — consumers see expired keys as fresh",
                "replica lag normal; staleness originates here",
            ]
        infra.propagate()

    if is_lag:
        root = "The read replica fell behind the primary; failover/resync it."
        fix = [("failover", "db-replica")]
    else:
        root = (
            "Cache invalidation wedged and the cache is serving expired keys — the replica is "
            "healthy; restart the cache."
        )
        fix = [("restart", "cache")]
    return IncidentSpec(
        "stale_reads",
        "Users seeing stale data",
        root,
        "PAGE: users seeing stale data, read latency climbing across api",
        fix,
        apply,
    )


def _latency_surge(rng: Random) -> IncidentSpec:
    # p99 collapses on a front-tier node. Half the time it is a genuine traffic surge (scale out);
    # half the time the node is thrashing GC on a leaked heap and scaling just spreads sick
    # replicas (restart). RPS vs memory dials disambiguate in one inspect.
    node = rng.choice(("web", "api"))
    p99 = round(rng.uniform(900.0, 3000.0), 1)
    mem = round(rng.uniform(93.0, 99.0), 1)
    is_surge = rng.random() < 0.5

    def apply(infra: Infra) -> None:
        s = infra.nodes[node]
        s.status, s.incident = "degraded", "latency_surge"
        s.metrics["latency_ms"] = p99
        if is_surge:
            s.metrics["rps_x_baseline"] = 2.4
            s.logs = ["RPS 2.4x baseline, connection queue saturating", "memory and GC normal"]
        else:
            s.metrics["mem_pct"] = mem
            s.logs = ["GC pause 900ms+, heap near limit", "RPS at baseline — load is NOT elevated"]
        infra.propagate()

    if is_surge:
        root = "A real traffic surge overwhelmed the front tier; scale it out — nothing is broken."
        fix = [("scale", node)]
    else:
        root = (
            "Not load: a leaked heap has the node thrashing GC; scaling spreads sick replicas — "
            "restart the node to reclaim memory."
        )
        fix = [("restart", node)]
    return IncidentSpec(
        "latency_surge",
        "p99 collapsing on the front tier",
        root,
        f"PAGE: {node} p99 {p99}ms, error budget burning",
        fix,
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
    Fault(
        "deploy_regression",
        "Errors spiking after a deploy",
        "Same page, two roots: a real regression (rollback) or a pool leak the deploy surfaced "
        "(restart) — the logs discriminate.",
        _deploy_regression,
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
    Fault(
        "stale_reads",
        "Users seeing stale data",
        "Same page, two roots: replica lag (failover) or wedged cache invalidation (restart "
        "cache) — replica lag settles it in one inspect.",
        _stale_reads,
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
    Fault(
        "latency_surge",
        "p99 collapsing on the front tier",
        "Same page, two roots: an organic surge (scale) or GC thrash on a leaked heap (restart) — "
        "the RPS and memory dials discriminate.",
        _latency_surge,
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
