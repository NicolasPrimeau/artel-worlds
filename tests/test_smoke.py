from fastapi.testclient import TestClient

from worlds.agent import HeuristicAgent
from worlds.config import DEFAULT
from worlds.genome import random_genome
from worlds.tick import step
from worlds.world import World


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
    from worlds import server

    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        state = client.get("/state").json()
        assert state["population"] > 0
        joined = client.post("/join", json={"agent_id": "test-agent"}).json()
        oid = joined["organism_id"]
        assert client.get(f"/perceive/{oid}").status_code == 200
        assert client.post(f"/intend/{oid}", json={"verb": "metabolize"}).json()["ok"] is True
