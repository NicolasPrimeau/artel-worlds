from phalanx.arena import Arena
from phalanx.config import DEFAULT
from phalanx.control import Bot


def _bots(arena, coordinated=()):
    boards, bots = {}, {}
    for t in arena.tanks.values():
        coord = t.team in coordinated
        board = boards.setdefault(t.team, {}) if coord else {}
        bots[t.id] = Bot(t.id, t.team, board, coord)
    return bots


def _run(arena, ticks, coordinated=()):
    bots = _bots(arena, coordinated)
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
    shooter.q, shooter.r, shooter.cooldown = 5, 5, 0
    victim.q, victim.r = 8, 5  # 3 hexes away, clear line of sight
    v0 = victim.energy
    a.submit(shooter.id, {"fire": victim.id})
    a.step()
    assert a.tanks[victim.id].energy <= v0 - DEFAULT.shot_damage + 1  # target took the hit
    assert shooter.cooldown > 0  # gun went on cooldown


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
    a.submit(shooter.id, {"fire": victim.id})
    a.step()
    assert a.tanks[victim.id].energy >= v0  # shot blocked: no damage, only regen


def test_shared_board_spreads_one_tanks_sighting_to_the_team():
    a = Arena(DEFAULT, seed=4)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (12, 9))
    bots = _bots(a, coordinated=("artel",))
    artel = a.team_tanks("artel")
    red = a.team_tanks("red")
    # drop one scout right next to one red so exactly one teammate sees it
    scout, far = artel[0], artel[1]
    scout.q, scout.r = red[0].q - 1, red[0].r
    far.q, far.r = 2, 2
    a.walls.clear()
    for t in artel:
        bots[t.id].decide(a.perceive(t.id), a.cfg, a.tick_count)
    assert red[0].id in bots[far.id].board  # the scout's sighting reached the whole team


def test_red_boards_stay_private():
    a = Arena(DEFAULT, seed=4)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (12, 9))
    bots = _bots(a)  # nobody coordinated: every board private
    artel = a.team_tanks("artel")
    target = a.team_tanks("red")[0]
    scout, far = artel[0], artel[1]
    scout.q, scout.r = target.q - 1, target.r
    far.q, far.r = 2, 2
    a.walls.clear()
    for t in artel:
        bots[t.id].decide(a.perceive(t.id), a.cfg, a.tick_count)
    assert target.id in bots[scout.id].board
    assert target.id not in bots[far.id].board  # no sharing without coordination
