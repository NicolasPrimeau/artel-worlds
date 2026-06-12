import os
import random
import tempfile

from watchtower.config import DEFAULT, MAX_ACTIONS_PER_INCIDENT, UNRESOLVED_MTTR
from watchtower.faults import FAMILIES, family_keys
from watchtower.incidents import Incident, make_stream, spec_for
from watchtower.infra import Infra
from watchtower.metrics import Metrics


def _fresh_incident(spec, fleet="artel"):
    return Incident(spec, 0, Infra(DEFAULT), fleet)


def test_stream_is_deterministic_and_paired():
    a = make_stream(123, 30)
    b = make_stream(123, 30)
    assert [s.family for s in a] == [s.family for s in b]
    assert [s.alert for s in a] == [s.alert for s in b]
    one = spec_for(123, 7)
    two = spec_for(123, 7)
    assert one.family == two.family and one.alert == two.alert and one.fix == two.fix


def test_same_spec_perturbs_both_fleets_identically():
    # the paired-trial guarantee: one spec applied to two infras must yield byte-identical state
    spec = spec_for(999, 3)
    ia, ib = Infra(DEFAULT), Infra(DEFAULT)
    spec.apply(ia)
    spec.apply(ib)
    for name in DEFAULT.node_names:
        assert ia.nodes[name].status == ib.nodes[name].status
        assert ia.nodes[name].metrics == ib.nodes[name].metrics
        assert ia.nodes[name].logs == ib.nodes[name].logs


def test_every_family_resolves_with_its_fix():
    for fault in FAMILIES:
        spec = fault.spawn(random.Random(fault.key))
        inc = _fresh_incident(spec)
        for action, node in spec.fix:
            inc.act(action, node)
        assert inc.resolved, f"{fault.key} did not resolve on its own fix sequence"
        assert inc.infra.all_healthy(), f"{fault.key} left the graph dirty after fix"


def test_runbook_path_beats_blind_path():
    # the whole thesis: knowing the fix (runbook) costs far less than discovering it (blind).
    spec = spec_for(42, 1)
    runbook = _fresh_incident(spec)
    for action, node in spec.fix:
        runbook.act(action, node)

    blind = _fresh_incident(spec)
    for n in ("api", "web", "db", "cache"):  # flail across the loud nodes first
        blind.act("inspect", n)
    blind.act("restart", "web")  # a wrong remediation
    blind.act("rollback", "api")  # another
    for action, node in spec.fix:
        blind.act(action, node)

    assert runbook.resolved and blind.resolved
    assert runbook.mttr() < blind.mttr()


def test_unresolved_incident_is_capped_and_booked_as_miss():
    spec = spec_for(7, 2)
    inc = _fresh_incident(spec)
    for _ in range(MAX_ACTIONS_PER_INCIDENT + 2):
        inc.act("inspect", "lb")  # never the fix
    assert inc.missed
    assert inc.mttr() == UNRESOLVED_MTTR
    assert inc.infra.all_healthy()  # world auto-remediated so the stream isn't blocked


def test_dependency_propagation_ripples_up():
    a = Infra(DEFAULT)
    a.nodes["db"].status = "down"
    a.nodes["db"].incident = "x"
    a.propagate()
    # api, auth, web, lb all lean on db transitively -> degraded
    assert a.nodes["api"].status == "degraded"
    assert a.nodes["web"].status == "degraded"
    assert a.nodes["lb"].status == "degraded"


def test_wrong_node_remediation_has_no_effect():
    spec = next(f for f in FAMILIES if f.key == "cache_down").spawn(random.Random(1))
    inc = _fresh_incident(spec)
    out = inc.act("restart", "db")  # right action, wrong node
    assert "no effect" in out["result"]
    assert not inc.resolved


def test_metrics_wedge_summary_and_pairing():
    with tempfile.TemporaryDirectory() as d:
        m = Metrics(os.path.join(d, "t.db"))
        for seq in range(20):
            m.record(seq, "disk_full", "artel", 60.0 + seq, 3, 1)
            m.record(seq, "disk_full", "solo", 200.0, 6, 1)
        s = m.summary()
        assert s["incidents"] == 20
        assert s["artel_mttr_all"] < s["solo_mttr_all"]
        assert s["artel_win_rate"] == 1.0
        w = m.wedge(bucket=10)
        assert len(w) == 2 and w[0]["artel_mttr"] is not None
        assert len(m.recent(5)) == 5
        m.close()


def test_families_count():
    assert len(family_keys()) == 12
    assert len(set(family_keys())) == 12


def test_server_boots_without_llm(monkeypatch):
    monkeypatch.setenv("WATCHTOWER_DB", tempfile.mktemp(suffix=".db"))
    from fastapi.testclient import TestClient

    import watchtower.server as server

    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        snap = client.get("/state").json()
        assert len(snap["artel_wall"]) == 10 and len(snap["solo_wall"]) == 10
        assert client.post("/fire").json()["fired"] is False  # disabled: no keys
