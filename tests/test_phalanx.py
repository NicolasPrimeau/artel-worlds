from phalanx.arena import Arena
from phalanx.config import DEFAULT
from phalanx.control import decide


def _run(arena, ticks):
    for _ in range(ticks):
        for t in list(arena.tanks.values()):
            p = arena.perceive(t.id)
            if p:
                arena.submit(t.id, decide(p, arena.cfg))
        arena.step()


def test_arena_seeds_and_combat_resolves():
    a = Arena(DEFAULT, seed=1)
    a.seed_house()
    start = len(a.tanks)
    assert start == DEFAULT.house_teams * DEFAULT.team_size
    _run(a, 400)
    assert a.tick_count == 400
    # convergent combat: tanks died and/or a team won outright
    assert len(a.tanks) < start or a.winner is not None


def test_perceive_is_fog_of_war():
    a = Arena(DEFAULT, seed=2)
    a.seed_house()
    tid = next(iter(a.tanks))
    p = a.perceive(tid)
    assert {"x", "y", "heading", "gun_heading", "energy", "gun_ready", "visible"} <= set(p)
    for v in p["visible"]:
        assert v["dist"] <= DEFAULT.sensor_range
        # enemy energy hidden unless close; allies always shown
        if v["kind"] == "enemy" and v["dist"] > DEFAULT.sensor_range // 2:
            assert "energy" not in v


def test_firing_costs_and_damages():
    a = Arena(DEFAULT, seed=3)
    # two enemies face to face, one shot lined up
    a.add_team("blue", "house:blue", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    shooter = next(t for t in a.tanks.values() if t.team == "blue")
    victim = next(t for t in a.tanks.values() if t.team == "red")
    shooter.x, shooter.y, shooter.gun, shooter.cooldown = 5, 5, 2, 0  # gun = East
    victim.x, victim.y = 8, 5  # 3 cells east, on the gun ray
    e0 = shooter.energy
    v0 = victim.energy
    a.submit(shooter.id, {"fire": 2.0})
    a.step()
    assert shooter.energy < e0  # paid for the shot
    a.step()  # shell travels and connects
    assert a.tanks[victim.id].energy < v0  # took damage
