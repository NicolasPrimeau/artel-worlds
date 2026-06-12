from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT, Config


# Healthy resting metrics — the numbers an inspect returns when nothing is wrong. Faults push
# specific dials into the red; a resolution returns the node to these. The point of the baseline
# is contrast: an agent reading metrics has to tell a real anomaly from ordinary noise.
BASELINE = {
    "cpu_pct": 22.0,
    "mem_pct": 41.0,
    "latency_ms": 38.0,
    "error_rate": 0.2,
    "queue_depth": 4.0,
    "disk_pct": 54.0,
    "conns": 24.0,
    "replica_lag_s": 0.3,
}


@dataclass
class NodeState:
    name: str
    kind: str
    status: str = "healthy"  # healthy | degraded | down
    metrics: dict = field(default_factory=lambda: dict(BASELINE))
    deploy_version: int = 7
    incident: str | None = None  # fault family key currently afflicting THIS node (root cause)
    logs: list = field(default_factory=list)


class Infra:
    """One fleet's copy of the service graph. The shared incident stream perturbs both fleets'
    Infra identically; each fleet's responder heals its own. The status wall renders this state,
    so a slower fleet visibly carries red nodes longer — the divergence made literal on screen."""

    def __init__(self, cfg: Config = DEFAULT):
        self.cfg = cfg
        self.nodes: dict[str, NodeState] = {n.name: NodeState(n.name, n.kind) for n in cfg.nodes}

    def reset(self) -> None:
        for n in self.cfg.nodes:
            s = self.nodes[n.name]
            s.status, s.incident = "healthy", None
            s.metrics = dict(BASELINE)
            s.logs = []

    def heal(self, name: str) -> None:
        # a node returns to rest: its own fault cleared and its dials back to baseline. Propagation
        # is recomputed by the caller so dependents that were only degraded-by-association recover.
        s = self.nodes.get(name)
        if s is None:
            return
        s.status, s.incident = "healthy", None
        s.metrics = dict(BASELINE)
        s.logs = []

    def propagate(self) -> None:
        # a node leaning on a down/degraded dependency shows degraded too — UNLESS it carries its
        # own root-cause incident (that's worse and stays). This is why the loudest alarm misleads:
        # 'api' screams 5xx when the real fault is 'db' two hops down. Iterate to a fixed point so a
        # failure ripples the whole depth of the spine.
        for _ in range(len(self.cfg.nodes)):
            changed = False
            for node in self.cfg.nodes:
                s = self.nodes[node.name]
                if s.incident is not None:
                    continue
                sick_dep = any(
                    self.nodes[d].status in ("down", "degraded") for d in node.depends_on
                )
                want = "degraded" if sick_dep else "healthy"
                if s.status != want:
                    s.status = want
                    if want == "degraded":
                        s.metrics["error_rate"] = max(s.metrics["error_rate"], 6.0)
                        s.metrics["latency_ms"] = max(s.metrics["latency_ms"], 220.0)
                    else:
                        s.metrics = dict(BASELINE)
                    changed = True
            if not changed:
                break

    def alarms(self) -> list[str]:
        # what the monitoring board is lit up about right now, root causes and ripples alike
        out = []
        for node in self.cfg.nodes:
            s = self.nodes[node.name]
            if s.status == "down":
                out.append(f"{s.name} DOWN")
            elif s.status == "degraded":
                out.append(f"{s.name} degraded")
        return out

    def inspect(self, name: str) -> dict:
        s = self.nodes.get(name)
        if s is None:
            return {"error": f"no node named {name}"}
        node = self.cfg.node_by_name[name]
        hot = {k: round(v, 1) for k, v in s.metrics.items() if v != BASELINE[k]}
        return {
            "node": s.name,
            "kind": s.kind,
            "status": s.status,
            "depends_on": list(node.depends_on),
            "deploy_version": s.deploy_version,
            "metrics": {k: round(v, 1) for k, v in s.metrics.items()},
            "anomalies": hot or "none",
        }

    def read_logs(self, name: str) -> dict:
        s = self.nodes.get(name)
        if s is None:
            return {"error": f"no node named {name}"}
        return {"node": s.name, "logs": list(s.logs) or ["(nothing unusual in the last window)"]}

    def status_wall(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "kind": s.kind,
                "status": s.status,
                "incident": s.incident,
            }
            for s in (self.nodes[n.name] for n in self.cfg.nodes)
        ]

    def all_healthy(self) -> bool:
        return all(s.status == "healthy" for s in self.nodes.values())
