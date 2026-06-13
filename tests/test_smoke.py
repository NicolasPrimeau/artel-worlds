import random

from fastapi.testclient import TestClient

from automata.agent import HeuristicAgent
from automata.config import DEFAULT
from automata.genome import crossover, mutate, random_genome
from automata.tick import step
from automata.world import World


def test_genomes_never_have_duplicate_genes():
    rng = random.Random(0)
    pool = [random_genome(rng, DEFAULT.max_genes) for _ in range(40)]
    for _ in range(3000):
        g = rng.choice(pool)
        g = (
            mutate(g, rng, DEFAULT)
            if rng.random() < 0.5
            else crossover(g, rng.choice(pool), rng, DEFAULT.max_genes)
        )
        pool.append(g)
    for g in pool:
        assert len(g.behaviors) == len(set(g.behaviors)), g.behaviors


def test_world_seeds_and_ticks():
    w = World(DEFAULT, seed=1)
    w.seed(DEFAULT.initial_population)
    assert w.stats()["population"] == DEFAULT.initial_population
    for _ in range(20):
        step(w, HeuristicAgent())
    assert w.tick_count == 20
    assert w.stats()["population"] > 0


def test_perceive_submit_contract():
    w = World(DEFAULT, seed=2)
    w.seed(10)
    org_id = next(iter(w.organisms))
    view = w.perceive(org_id)
    assert view is not None
    assert "my_energy" in view
    assert w.submit(org_id, "metabolize", "random") is True
    assert w.submit(999999, "metabolize", "random") is False


def test_remote_organism_uses_same_path():
    w = World(DEFAULT, seed=3)
    w.seed(5)
    cell = next(c for c in w.cells.values() if c.organism is None)
    g = random_genome(w.rng, DEFAULT.max_genes)
    org = w.spawn(cell.q, cell.r, g, w.new_lineage(), DEFAULT.birth_energy, agent_id="remote-1")
    w.submit(org.id, "metabolize", "random")
    step(w, HeuristicAgent())
    assert w.tick_count == 1


def test_server_endpoints():
    from automata import server

    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        state = client.get("/state").json()
        assert state["population"] > 0
        joined = client.post("/join", json={"agent_id": "test-agent"}).json()
        oid = joined["organism_id"]
        assert client.get(f"/perceive/{oid}").status_code == 200
        assert client.post(f"/intend/{oid}", json={"verb": "metabolize"}).json()["ok"] is True
        detail = client.get(f"/organism/{oid}").json()
        assert detail["agent"] == "test-agent"
        assert "behaviors" in detail["genome"]


def test_tribe_play():
    from automata import server

    with TestClient(server.app) as client:
        joined = client.post("/join", json={"agent_id": "chieftain"}).json()
        tribe, token = joined["tribe"], joined["token"]
        hdr = {"Authorization": f"Bearer {token}"}
        assert len(joined["organisms"]) >= 1
        bundle = client.get(f"/tribe/{tribe}/perceive", headers=hdr).json()
        assert bundle["controller"] == "chieftain"
        assert set(bundle["members"]) == {str(i) for i in joined["organisms"]}
        ids = list(bundle["members"])
        actions = {i: {"verb": "migrate", "target": "toxin_min"} for i in ids}
        applied = client.post(
            f"/tribe/{tribe}/intend", json={"actions": actions}, headers=hdr
        ).json()
        assert applied["applied"] == len(ids)
        # no token, or the wrong one, can't touch the tribe
        assert client.get(f"/tribe/{tribe}/perceive").status_code == 401
        assert (
            client.get(
                f"/tribe/{tribe}/perceive", headers={"Authorization": "Bearer nope"}
            ).status_code
            == 401
        )
        assert client.get("/tribe/999999/perceive", headers=hdr).status_code == 404


def test_agent_card_and_playbook():
    from automata import server

    with TestClient(server.app) as client:
        card = client.get("/card").json()
        assert card["actions"]["verbs"] and card["perception"] and card["endpoints"]
        assert "metabolize" in card["actions"]["verbs"]
        txt = client.get("/llms.txt")
        assert txt.status_code == 200
        assert "/join" in txt.text and "perceive" in txt.text


def test_automata_claude_sdk_client_meters_spend(monkeypatch):
    import asyncio

    import claude_agent_sdk as sdk

    from automata.llm import ClaudeSDKClient

    async def fake_query(prompt, options):
        assert options.max_turns == 1 and "tribe" in options.system_prompt
        from unittest.mock import MagicMock

        msg = MagicMock(spec=sdk.ResultMessage)
        msg.is_error = False
        msg.result = '{"regulators": {}, "behaviors": []}'
        msg.total_cost_usd = 0.002
        yield msg

    monkeypatch.setattr(sdk, "query", fake_query)
    c = ClaudeSDKClient("haiku")
    out = asyncio.run(c.complete("you are a tribe", "rewrite"))
    assert out.startswith('{"regulators"')
    assert c.spent == 0.002
