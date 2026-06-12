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


def test_metrics_history_pairs_and_orders():
    with tempfile.TemporaryDirectory() as d:
        m = Metrics(os.path.join(d, "t.db"))
        for seq in range(5):
            m.record(seq, "traffic_spike", "artel", 100 - seq * 10, 3, True)
            m.record(seq, "traffic_spike", "solo", 120, 5, True)
        m.record(9, "bad_deploy", "artel", 50, 2, True)  # unpaired: solo row missing

        hist = m.history()
        assert [h["seq"] for h in hist] == [0, 1, 2, 3, 4]  # paired only, oldest first
        assert hist[0]["artel"] == 100 and hist[0]["solo"] == 120
        assert hist[-1]["artel"] == 60
        assert all(h["family"] == "traffic_spike" for h in hist)
        m.close()


def test_resolved_incident_always_records_a_runbook():
    # the resolving action ends the respond() loop before the model's RECORD step — the
    # post-resolution round (with deterministic fallback) must guarantee a runbook anyway
    import asyncio

    from watchtower import agent as A

    spec = spec_for(123, 0)
    inc = _fresh_incident(spec)
    for action, node in spec.fix:
        inc.act(action, node)
    assert inc.resolved

    store = A.SoloStore("solo-test")

    async def boom(http, system, transcript):
        raise RuntimeError("llm down")

    orig = A._chat
    A._chat = boom
    try:
        asyncio.run(A._record_runbook(None, inc, store, [], None))
    finally:
        A._chat = orig
    assert len(store.notes) == 1
    assert spec.family in store.notes[0]["text"]
    fix_step = f"{spec.fix[0][0]} {spec.fix[0][1]}"
    assert fix_step in store.notes[0]["text"]


def test_solo_board_is_private_and_tracks_incident_lifecycle():
    import asyncio

    from watchtower import agent as A

    async def run():
        s1 = A.SoloStore("solo-1")
        s2 = A.SoloStore("solo-2")
        tid = await s1.open_incident(4, "Replica lag", "db_primary_stuck", "alert text")
        assert "Incident #4" in await s1.board()
        assert await s2.board() == ""  # private: a teammate's board shows nothing

        await s1.close_incident(tid, resolved=False, note="gave up")
        assert "gave up" in await s1.board()  # miss stays open, note attached

        await s1.file_task("add lag alerting", "threshold gap")
        assert await s1.finish_task("t1") is True
        assert "lag alerting" not in await s1.board()

        await s1.sweep_family("db_primary_stuck")
        assert "Incident #4" not in await s1.board()  # family cracked: old incident closed

    asyncio.run(run())


def test_handoff_is_private_for_solo_fleet():
    import asyncio

    from watchtower import agent as A

    async def run():
        s1 = A.SoloStore("solo-1")
        s2 = A.SoloStore("solo-2")
        await s1.save_handoff("incident #3 (db_primary_stuck): resolved in 240s")
        assert "incident #3" in await s1.handoff()  # own last shift carries over
        assert await s2.handoff() == ""  # a teammate inherits nothing — no delta, no team

    asyncio.run(run())


def _spawn_branch(key, want_fix_action, tries=400):
    from random import Random

    for i in range(tries):
        f = next(f for f in FAMILIES if f.key == key)
        spec = f.spawn(Random(i))
        if spec.fix[0][0] == want_fix_action:
            return spec
    raise AssertionError(f"{key}: no spawn with fix {want_fix_action} in {tries} tries")


def test_branched_families_draw_both_roots():
    # probabilistic graphs: the same presentation must occur with each of its roots
    for key, fixes in (
        ("deploy_regression", ("rollback", "restart")),
        ("stale_reads", ("failover", "restart")),
        ("latency_surge", ("scale", "restart")),
    ):
        for fix in fixes:
            assert _spawn_branch(key, fix)


def test_branched_alert_never_leaks_the_root():
    # the pager text must be identical in SHAPE across roots — diagnosis, not the alert,
    # discriminates. Strip digits/node names and the templates must match.
    import re

    def shape(s):
        s = re.sub(r"[0-9.]+", "#", s)
        return re.sub(r"\b(api|web|auth|lb)\b", "@", s)

    for key in ("deploy_regression", "stale_reads", "latency_surge"):
        fixes = {
            "deploy_regression": ("rollback", "restart"),
            "stale_reads": ("failover", "restart"),
            "latency_surge": ("scale", "restart"),
        }[key]
        a = _spawn_branch(key, fixes[0])
        b = _spawn_branch(key, fixes[1])
        assert shape(a.alert) == shape(b.alert)


def test_wrong_branch_fix_has_no_effect_but_right_one_resolves():
    # the reflex fix on the WRONG root burns time and changes nothing; a solo agent can still
    # diagnose and recover — stumbling, not stonewalled
    spec = _spawn_branch("deploy_regression", "restart")  # pool-leak root: rollback is the reflex
    inc = _fresh_incident(spec)
    node = spec.fix[0][1]
    out = inc.act("rollback", node)
    assert not inc.resolved and "no effect" in str(out.get("result", ""))
    out = inc.act("restart", node)
    assert inc.resolved and out.get("resolved") is True


def test_branched_logs_carry_the_discriminator():
    # a careful agent must be able to settle the branch from diagnostics alone
    reg = _spawn_branch("deploy_regression", "rollback")
    pool = _spawn_branch("deploy_regression", "restart")
    ia, ib = Infra(DEFAULT), Infra(DEFAULT)
    reg.apply(ia)
    pool.apply(ib)
    reg_logs = " ".join(ia.nodes[reg.fix[0][1]].logs)
    pool_logs = " ".join(ib.nodes[pool.fix[0][1]].logs)
    assert "NEW request handler" in reg_logs
    assert "pool" in pool_logs and "no errors attributable to new code" in pool_logs


def test_branched_apply_is_pure_across_fleets():
    # paired-trial guarantee must hold for branched specs too: one spec, two infras, identical state
    for key, fix in (
        ("deploy_regression", "restart"),
        ("latency_surge", "restart"),
        ("stale_reads", "restart"),
    ):
        spec = _spawn_branch(key, fix)
        ia, ib = Infra(DEFAULT), Infra(DEFAULT)
        spec.apply(ia)
        spec.apply(ib)
        for name in DEFAULT.node_names:
            assert ia.nodes[name].status == ib.nodes[name].status
            assert ia.nodes[name].metrics == ib.nodes[name].metrics
            assert ia.nodes[name].logs == ib.nodes[name].logs


def test_epochs_open_new_roots_over_time():
    # the world is non-stationary: a migration-wedge root exists at epoch 2+ but NEVER at epoch 0
    from random import Random

    fault = next(f for f in FAMILIES if f.key == "deploy_regression")
    early_fixes = {fault.spawn(Random(i), 0).fix[0] for i in range(300)}
    late_fixes = {fault.spawn(Random(i), 4).fix[0] for i in range(300)}
    assert ("restart", "db") not in early_fixes
    assert ("restart", "db") in late_fixes
    # and the original branches still occur late — new paths open, old ones don't vanish
    assert any(a == "rollback" for a, _ in late_fixes)


def test_epoch_stream_stays_paired():
    a = make_stream(77, 100)
    b = make_stream(77, 100)
    assert [s.fix for s in a] == [s.fix for s in b]


def test_solo_retention_fades_cold_paths_and_keeps_hot_ones():
    import asyncio

    from watchtower import agent as A

    async def run():
        s = A.SoloStore("solo-1")
        await s.remember("runbook stale_reads: replica lag high -> failover db-replica")
        await s.remember("runbook cert_expiry: handshake failures -> rotate lb")

        for _ in range(8):  # the cert runbook stays hot through repeated use
            assert "rotate lb" in await s.recall("certificate handshake failing")
            for _ in range(5):
                await s.save_handoff("shift")

        # 40 shifts later the unused replica runbook has faded past recall...
        assert "failover" not in await s.recall("stale data replica lag")
        # ...while the hot path is still there
        assert "rotate lb" in await s.recall("certificate handshake failing")

    asyncio.run(run())


def test_action_times_are_noisy_bounded_and_reproducible():
    spec = spec_for(42, 0)
    times = set()
    for fleet in ("artel", "solo"):
        inc = Incident(spec, 0, Infra(DEFAULT), fleet)
        inc.act("inspect", "db")
        times.add(inc.elapsed)
        assert 4.0 <= inc.elapsed <= 16.0  # clamped to [0.5x, 2x] of the 8s base
        again = Incident(spec, 0, Infra(DEFAULT), fleet)
        again.act("inspect", "db")
        assert again.elapsed == inc.elapsed  # seeded: same incident + fleet replays identically
    assert len(times) == 2  # but the two fleets draw independently


def test_kv_roundtrip_survives_reopen():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.db")
        m = Metrics(path)
        m.kv_set("spend", '{"total": 1.5}')
        m.kv_set("spend", '{"total": 2.5}')
        m.close()
        m2 = Metrics(path)
        assert m2.kv_get("spend") == '{"total": 2.5}'
        m2.kv_delete("spend")
        assert m2.kv_get("spend") is None
        m2.close()


def test_world_state_survives_a_restart(monkeypatch):
    # a deploy must not wipe the solo fleet's notebooks or the spend counters — that biases
    # the A/B (Artel's memory lives on artel.run and already survives)
    import asyncio

    from watchtower import agent as A
    from watchtower.world import Responder, World

    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("WATCHTOWER_DB", os.path.join(d, "w.db"))

        async def run():
            w = World()
            s = A.SoloStore("solo-1")
            await s.remember("runbook stale_reads: check replica lag first")
            await s.save_handoff("incident #2: resolved")
            w.solo = [Responder("solo-1", s)]
            w.spent_total, w.spent_today = 1.23, 0.04
            w._persist_state()
            await w.aclose()

            w2 = World()
            s2 = A.SoloStore("solo-1")
            w2.solo = [Responder("solo-1", s2)]
            w2._restore_state()
            assert s2.notes and "replica lag" in s2.notes[0]["text"]
            assert s2.shift == 1 and "incident #2" in s2.last_shift
            assert w2.spent_total == 1.23
            await w2.aclose()

        asyncio.run(run())


def test_failover_accepts_either_side_of_the_promotion():
    from watchtower.faults import _db_primary_stuck
    from random import Random

    spec = _db_primary_stuck(Random(5))
    inc = _fresh_incident(spec)
    out = inc.act("failover", "db-replica")  # the intuitive target: promote the replica
    assert "applied" in str(out.get("result", ""))
    inc.act("restart", "db")
    assert inc.resolved


def test_llm_outage_is_ridden_out_not_booked_as_a_miss(monkeypatch):
    import asyncio

    from watchtower import agent as A

    spec = spec_for(11, 0)
    inc = _fresh_incident(spec)
    store = A.SoloStore("solo-test")
    calls = {"n": 0}

    async def flaky(http, system, transcript):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 simulated")
        return "", [], 0, 0, {"cin": 0.0, "cout": 0.0}

    async def nosleep(_s):
        return None

    monkeypatch.setattr(A, "_chat", flaky)
    monkeypatch.setattr(A.asyncio, "sleep", nosleep)
    asyncio.run(A.respond(None, inc, store))
    assert calls["n"] == 3  # two failures absorbed, the third answered
