from phalanx.arena import Arena
from phalanx.config import DEFAULT
from phalanx.control import Bot


def _bots(arena):
    return {t.id: Bot(t.id, t.team) for t in arena.tanks.values()}


def _run(arena, ticks):
    bots = _bots(arena)
    for _ in range(ticks):
        for t in list(arena.tanks.values()):
            p = arena.perceive(t.id)
            if p:
                arena.submit(t.id, bots[t.id].decide(p, arena.cfg, arena.tick_count))
        arena.step()


def test_arena_seeds_and_combat_resolves():
    a = Arena(DEFAULT, seed=1)
    a.seed_house()
    start = len(a.tanks)
    assert start == DEFAULT.house_teams * DEFAULT.team_size
    _run(a, 400)
    # convergent combat: tanks died and/or a team won outright
    assert len(a.tanks) < start or a.winner is not None


def test_perceive_is_fog_of_war():
    a = Arena(DEFAULT, seed=2)
    a.seed_house()
    tid = next(iter(a.tanks))
    p = a.perceive(tid)
    assert {"q", "r", "heading", "energy", "gun_ready", "visible"} <= set(p)
    for v in p["visible"]:
        assert v["dist"] <= DEFAULT.sensor_range
        # enemy energy hidden unless close; allies always shown
        if v["kind"] == "enemy" and v["dist"] > DEFAULT.sensor_range // 2:
            assert "energy" not in v


def test_firing_hits_target_in_range_with_los():
    a = Arena(DEFAULT, seed=3)
    a.add_team("blue", "house:blue", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    shooter = next(t for t in a.tanks.values() if t.team == "blue")
    victim = next(t for t in a.tanks.values() if t.team == "red")
    a.walls.clear()
    for i, o in enumerate(t for t in a.tanks.values() if t not in (shooter, victim)):
        o.q, o.r = 1 + i, 12  # park bystanders off the firing line — tanks block shots now
    shooter.q, shooter.r, shooter.cooldown = 5, 5, 0
    victim.q, victim.r = 8, 5  # 3 hexes away, clear line of sight
    v0 = victim.energy
    a.submit(shooter.id, {"fire": victim.id})
    a.step()
    assert a.tanks[victim.id].energy <= v0 - DEFAULT.shot_damage + 1  # target took the hit
    assert shooter.cooldown == max(0, DEFAULT.gun_cooldown - 1)  # reload started, then ticked once


def test_walls_block_vision():
    a = Arena(DEFAULT, seed=9)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    me = next(t for t in a.tanks.values() if t.team == "artel")
    foe = next(t for t in a.tanks.values() if t.team == "red")
    a.walls.clear()
    me.q, me.r = 5, 5
    foe.q, foe.r = 8, 5  # 3 hexes away, within sensor range
    assert any(v["id"] == foe.id for v in a.perceive(me.id)["visible"])  # clear LOS: seen
    a.walls = {(6, 5), (7, 5)}  # drop a wall between them
    assert all(v["id"] != foe.id for v in a.perceive(me.id)["visible"])  # now hidden


def test_wall_blocks_line_of_sight():
    a = Arena(DEFAULT, seed=5)
    a.add_team("blue", "house:blue", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    shooter = next(t for t in a.tanks.values() if t.team == "blue")
    victim = next(t for t in a.tanks.values() if t.team == "red")
    shooter.q, shooter.r, shooter.cooldown = 5, 5, 0
    victim.q, victim.r = 8, 5
    a.walls = {(6, 5), (7, 5)}  # wall sits on the line between them
    v0 = victim.energy
    s0 = shooter.energy
    a.submit(shooter.id, {"fire": victim.id})
    a.step()
    assert a.tanks[victim.id].energy >= v0  # shot blocked: no damage, only regen
    assert shooter.energy <= s0 - DEFAULT.shot_cost  # the trigger pull still cost energy
    assert shooter.cooldown == max(0, DEFAULT.gun_cooldown - 1)  # and started the reload


def test_each_tank_board_is_private():
    # deterministic tanks NEVER share what they see — only the tank that personally spots an
    # enemy records it; a teammate that didn't see it knows nothing. The sole coordination in
    # Phalanx is the live Artel LLM squad, never the deterministic bots.
    a = Arena(DEFAULT, seed=4)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (12, 9))
    bots = _bots(a)
    artel = a.team_tanks("artel")
    target = a.team_tanks("red")[0]
    scout, far = artel[0], artel[1]
    scout.q, scout.r = target.q - 1, target.r
    far.q, far.r = 2, 2
    a.walls.clear()
    for t in artel:
        bots[t.id].decide(a.perceive(t.id), a.cfg, a.tick_count)
    assert target.id in bots[scout.id].board  # the scout that saw it remembers it
    assert target.id not in bots[far.id].board  # the teammate that didn't see it has nothing


def test_server_starts_and_ticks_without_squad():
    # the synchronous tick loop and lifespan come up cleanly with the LLM squad off
    # (no agent keys) — Artel falls back to solo bots and the arena never stalls.
    from fastapi.testclient import TestClient

    from phalanx import server

    with TestClient(server.app) as client:
        assert client.get("/health").json()["status"] == "ok"


def test_scoreboard_survives_a_restart(tmp_path, monkeypatch):
    # deploys must not zero the series: scores, match counter, history, and spend reload
    # from the state file; only the in-flight match and agent working state are ephemeral
    monkeypatch.setenv("PHALANX_STATE", str(tmp_path / "state.json"))
    from phalanx.server import Phalanx

    g = Phalanx()
    g.scores = {"artel": 7, "red": 4}
    g.history.append({"match": g.match_no, "winner": "artel"})
    g.squad.spent = 0.42
    g.persist_state()

    g2 = Phalanx()
    assert g2.scores == {"artel": 7, "red": 4}
    assert g2.history[-1]["winner"] == "artel"
    assert g2.squad.spent == 0.42
    assert g2.match_no == g.match_no + 1  # the wiped in-flight match restarts as the next one
