import asyncio


def test_app_builds_with_expected_routes():
    from vibequest import server

    paths = {r.path for r in server.app.routes}
    assert {"/play", "/ws", "/state"} <= paths


def test_should_auto_resolve_truth_table(monkeypatch):
    # R1: the DM auto-resolves only when ready + read-timer elapsed + under the idle streak cap.
    from vibequest import server

    now = 10_000.0
    elapsed = now - server.DECISION_TIMEOUT - 1.0

    monkeypatch.setattr(server, "_ready", {object()})
    monkeypatch.setattr(server, "_decision_at", elapsed)
    monkeypatch.setattr(server, "_auto_streak", 0)
    assert server._should_auto_resolve(now) is True

    # empty room -> never
    monkeypatch.setattr(server, "_ready", set())
    assert server._should_auto_resolve(now) is False

    # ready but the read-timer hasn't elapsed -> not yet
    monkeypatch.setattr(server, "_ready", {object()})
    monkeypatch.setattr(server, "_decision_at", now)
    assert server._should_auto_resolve(now) is False

    # ready + elapsed but the streak cap is hit -> paused (the cost fix: no runaway LLM spend)
    monkeypatch.setattr(server, "_decision_at", elapsed)
    monkeypatch.setattr(server, "_auto_streak", server.MAX_AUTO_STREAK)
    assert server._should_auto_resolve(now) is False


def test_broadcast_survives_client_join_mid_iteration(monkeypatch):
    # C1: a client connecting during a send must not raise "Set changed size during iteration".
    from vibequest import server

    class FakeWS:
        def __init__(self, on_send=None):
            self.sent = []
            self._on_send = on_send

        async def send_text(self, payload):
            if self._on_send:
                self._on_send()
            self.sent.append(payload)

    latecomer = FakeWS()
    first = FakeWS(on_send=lambda: server._clients.add(latecomer))
    monkeypatch.setattr(server, "_clients", {first})

    asyncio.run(server._broadcast({"type": "ping"}))  # must not raise
    assert first.sent


def test_broadcast_prunes_dead_clients(monkeypatch):
    from vibequest import server

    class DeadWS:
        async def send_text(self, payload):
            raise RuntimeError("socket closed")

    class LiveWS:
        def __init__(self):
            self.sent = []

        async def send_text(self, payload):
            self.sent.append(payload)

    dead, live = DeadWS(), LiveWS()
    monkeypatch.setattr(server, "_clients", {dead, live})

    asyncio.run(server._broadcast({"type": "ping"}))
    assert dead not in server._clients
    assert live in server._clients and live.sent


def test_narrate_fails_over_to_router_when_narration_empty(monkeypatch):
    from vibequest import llm

    class FakeRouter:
        def __init__(self, out):
            self.out = out
            self.calls = 0

        async def complete(self, req):
            self.calls += 1
            return self.out

    narr, pool = FakeRouter(""), FakeRouter("a beat from the pool")
    monkeypatch.setattr(llm, "NARRATION", narr)
    monkeypatch.setattr(llm, "ROUTER", pool)
    out = asyncio.run(llm._narrate(llm.Request(system="s", user="u")))
    assert out == "a beat from the pool"
    assert narr.calls == 1 and pool.calls == 1


def test_narrate_skips_router_when_narration_answers(monkeypatch):
    from vibequest import llm

    class FakeRouter:
        def __init__(self, out):
            self.out = out
            self.calls = 0

        async def complete(self, req):
            self.calls += 1
            return self.out

    narr, pool = FakeRouter("primary beat"), FakeRouter("unused")
    monkeypatch.setattr(llm, "NARRATION", narr)
    monkeypatch.setattr(llm, "ROUTER", pool)
    out = asyncio.run(llm._narrate(llm.Request(system="s", user="u")))
    assert out == "primary beat"
    assert pool.calls == 0  # no needless second call


def test_narrate_no_double_call_when_narration_is_router(monkeypatch):
    from vibequest import llm

    class FakeRouter:
        def __init__(self):
            self.calls = 0

        async def complete(self, req):
            self.calls += 1
            return ""

    shared = FakeRouter()
    monkeypatch.setattr(llm, "NARRATION", shared)
    monkeypatch.setattr(llm, "ROUTER", shared)
    out = asyncio.run(llm._narrate(llm.Request(system="s", user="u")))
    assert out == "" and shared.calls == 1  # same object -> don't call twice


def test_affinity_matches_card_type_to_obstacle():
    from vibequest import engine as e

    # a PERSON yields to social/confrontation; a THING to improvise/disrupt
    assert e.affinity_delta("boon", "person") == e.AFFINITY
    assert e.affinity_delta("rival", "person") == e.AFFINITY
    assert e.affinity_delta("twist", "person") == -e.AFFINITY
    assert e.affinity_delta("encounter", "person") == -e.AFFINITY
    # obstacle kind flips the affinities
    assert e.affinity_delta("twist", "thing") == e.AFFINITY
    assert e.affinity_delta("encounter", "thing") == e.AFFINITY
    assert e.affinity_delta("boon", "thing") == -e.AFFINITY
    assert e.obstacle_kind(True) == "person"
    assert e.obstacle_kind(False) == "thing"


def test_scene_progress_accumulates_across_rounds():
    # the progress bar now means something: apply_fit_effects no longer zeroes it each round,
    # so two strong plays fill it past the clear threshold (the server then breaks the wall).
    from vibequest.engine import QuestState, apply_fit_effects, SCENE_THRESHOLD

    q = QuestState(id="x", template_id="t", title="T", hook="h", complication="c")
    apply_fit_effects(q, 60)
    assert 0 < q.scene_progress < SCENE_THRESHOLD  # one strong play: partway
    apply_fit_effects(q, 60)
    assert q.scene_progress >= SCENE_THRESHOLD  # two strong plays: bar full -> wall can break


def test_tension_meter_removed():
    from vibequest.engine import QuestState

    assert not hasattr(
        QuestState(id="x", template_id="t", title="T", hook="h", complication="c"), "tension"
    )
