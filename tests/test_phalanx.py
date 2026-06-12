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

    g.completed = 11
    g.persist_state()

    g2 = Phalanx()
    assert g2.completed == 11
    assert g2.scores == {"artel": 7, "red": 4}
    assert g2.history[-1]["winner"] == "artel"
    assert g2.squad.spent == 0.42
    assert g2.match_no == g.match_no + 1  # the wiped in-flight match restarts as the next one


def test_bfs_routes_around_a_tank_when_a_detour_exists():
    from phalanx.tank import AXIAL_DIRS, bfs_step

    # straight-line corridor east with an ally parked one step ahead, open hexes around:
    # the path must detour, not queue
    R = 7
    me, target = (7, 7), (11, 7)
    ally = {(8, 7)}
    d = bfs_step(*me, *target, set(), R, soft=ally)
    assert d is not None
    step = (me[0] + AXIAL_DIRS[d][0], me[1] + AXIAL_DIRS[d][1])
    assert step not in ally  # routed around, not into/behind the ally


def test_bfs_queues_behind_a_tank_in_a_sole_corridor():
    from phalanx.tank import AXIAL_DIRS, bfs_step

    # walls force a single-file corridor: (8,7) is the only way east, an ally sits in it.
    # bfs returns the through-route's first step only via the queue fallback, which vetoes
    # step one — so the result is a WAIT (None) rather than an illegal shove
    R = 7
    walls = {(8, 6), (8, 8), (7, 6), (7, 8), (9, 6), (9, 8)}
    ally = {(8, 7)}
    d = bfs_step(7, 7, 11, 7, walls, R, soft=ally)
    assert d is None or (7 + AXIAL_DIRS[d][0], 7 + AXIAL_DIRS[d][1]) not in ally


def test_bfs_still_reaches_target_through_remembered_walls():
    from phalanx.tank import bfs_step

    # a wall pocket between us and the target: with the walls known, a path is found around
    R = 7
    walls = {(8, 6), (8, 7), (8, 8)}
    d = bfs_step(7, 7, 10, 7, walls, R)
    assert d is not None


def _duel(seed):
    a = Arena(DEFAULT, seed=seed)
    a.add_team("blue", "house:blue", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    shooter = next(t for t in a.tanks.values() if t.team == "blue")
    victim = next(t for t in a.tanks.values() if t.team == "red")
    a.walls.clear()
    for i, o in enumerate(t for t in a.tanks.values() if t not in (shooter, victim)):
        o.q, o.r = 1 + i, 12  # bystanders off the firing line
    shooter.q, shooter.r, shooter.cooldown = 5, 5, 0
    return a, shooter, victim


def test_shot_fires_from_turn_start_position_while_scooting():
    from phalanx.tank import AXIAL_DIRS

    a, shooter, victim = _duel(21)
    victim.q, victim.r = 8, 5
    shooter.heading = AXIAL_DIRS.index((0, -1))  # displace off the line after firing
    v0 = victim.energy
    a.submit(shooter.id, {"fire": victim.id, "power": 2, "move": "fwd"})
    a.step()
    assert (shooter.q, shooter.r) == (5, 4)  # the hull moved...
    assert a.tanks[victim.id].energy <= v0 - DEFAULT.shot_damage + 1  # ...the muzzle did not


def test_breaking_range_with_a_move_dodges_the_shot():
    from phalanx.tank import AXIAL_DIRS

    a, shooter, victim = _duel(22)
    victim.q, victim.r = 10, 5  # distance 5 — the edge of power-2 range
    victim.heading = AXIAL_DIRS.index((1, 0))
    v0, s0 = victim.energy, shooter.energy
    a.submit(shooter.id, {"fire": victim.id, "power": 2})
    a.submit(victim.id, {"move": "fwd"})  # steps to (11,5): distance 6, out of range
    a.step()
    assert a.tanks[victim.id].energy >= v0 - 1  # dodged: no damage (only move cost applies)
    assert shooter.energy <= s0 - DEFAULT.power_cost[1]  # the trigger pull still cost power-2
    assert shooter.cooldown == max(0, DEFAULT.gun_cooldown - 1)  # reload started, ticked once


def test_own_post_move_hull_does_not_block_own_shot():
    from phalanx.tank import AXIAL_DIRS

    a, shooter, victim = _duel(23)
    victim.q, victim.r = 8, 5
    shooter.heading = AXIAL_DIRS.index((1, 0))  # advance ALONG the firing line
    v0 = victim.energy
    a.submit(shooter.id, {"fire": victim.id, "power": 2, "move": "fwd"})
    a.step()
    assert (shooter.q, shooter.r) == (6, 5)  # now standing on the old firing line
    assert a.tanks[victim.id].energy <= v0 - DEFAULT.shot_damage + 1  # shot was not self-blocked


def test_operator_reset_wipes_the_series_but_keeps_spend(tmp_path, monkeypatch):
    monkeypatch.setenv("PHALANX_STATE", str(tmp_path / "state.json"))
    from fastapi.testclient import TestClient

    from phalanx import server

    with TestClient(server.app) as client:
        server.G.scores = {"artel": 5, "red": 3}
        server.G.completed = 8
        server.G.history.append({"match": 1, "winner": "artel"})
        server.G.squad.spent = 1.25
        server.G.persist_state()

        assert client.post("/reset").json()["ok"]
        assert server.G.scores == {} and server.G.completed == 0 and server.G.history == []
        assert server.G.squad.spent == 1.25  # money spent stays spent

        g2 = server.Phalanx()  # the wipe was persisted, not just in-memory
        assert g2.scores == {} and g2.completed == 0


def test_ballistic_miss_when_target_steps_off_the_line():
    from phalanx.tank import AXIAL_DIRS

    a, shooter, victim = _duel(31)
    victim.q, victim.r = 8, 5  # straight east of the shooter
    victim.heading = AXIAL_DIRS.index((0, -1))  # steps off the firing line
    v0 = victim.energy
    a.submit(shooter.id, {"fire": victim.id, "power": 2})
    a.submit(victim.id, {"move": "fwd"})
    a.step()
    assert a.tanks[victim.id].energy >= v0 - 1  # missed (only move cost)
    assert "MISSED" in shooter.last_fire or "cover" in shooter.last_fire
    assert any(tc.get("kind") != "hit" for tc in a.tracers)  # the miss is visible


def test_predictive_fire_at_leads_a_mover():
    from phalanx.tank import AXIAL_DIRS

    a, shooter, victim = _duel(32)
    victim.q, victim.r = 8, 5
    victim.heading = AXIAL_DIRS.index((0, -1))  # will step to (8,4)
    v0 = victim.energy
    a.submit(shooter.id, {"fire_at": [8, 4], "power": 2})  # aim where it is GOING
    a.submit(victim.id, {"move": "fwd"})
    a.step()
    assert a.tanks[victim.id].energy <= v0 - DEFAULT.shot_damage + 1
    assert shooter.last_fire == f"hit #{victim.id}"


def test_teammate_on_the_line_eats_the_shot():
    a, shooter, victim = _duel(33)
    mate = next(t for t in a.tanks.values() if t.team == shooter.team and t.id != shooter.id)
    victim.q, victim.r = 8, 5
    mate.q, mate.r = 6, 5  # parked on the firing line
    m0, v0 = mate.energy, victim.energy
    a.submit(shooter.id, {"fire": victim.id, "power": 2})
    a.step()
    assert a.tanks[victim.id].energy >= v0 - 1  # screened
    assert a.tanks[mate.id].energy <= m0 - DEFAULT.shot_damage + 1  # the teammate ate it
    assert "TEAMMATE" in shooter.last_fire
    assert shooter.energy < 100  # no reward for friendly hits, cost still paid


def test_fire_at_through_a_teammate_is_floored(monkeypatch):
    import asyncio

    from phalanx import agent as A

    p = {
        "id": 1,
        "q": 5,
        "r": 5,
        "heading": 0,
        "energy": 80,
        "gun_ready": True,
        "tick": 3,
        "width": 15,
        "height": 15,
        "map_radius": 7,
        "power_range": [3, 5, 7],
        "power_cost": [0, 2, 4],
        "fire_range": 7,
        "visible": [
            {"id": 2, "kind": "ally", "dq": 2, "dr": 0, "dist": 2, "dir": 0},
            {"id": 5, "kind": "enemy", "dq": 4, "dr": 0, "dist": 4, "dir": 0, "clear_shot": False},
        ],
        "walls": [],
        "safe": True,
        "zone_radius": 14,
        "dist_center": 2,
        "to_center": 0,
        "hit_taken": 0,
        "last_fire": "",
    }

    async def fake_chat(http, system, transcript, force=None, toolset=None):
        return (
            "",
            [{"id": "c1", "name": "act", "input": {"fire_at": [9, 5], "power": 3}}],
            0,
            0,
            {"cin": 0.0, "cout": 0.0},
        )

    monkeypatch.setattr(A, "_chat", fake_chat)
    intent, cost, plan, inbox, recalled = asyncio.run(
        A.decide(None, {"id": "t", "key": "k"}, p, [], solo=True)
    )
    assert "fire_at" not in intent  # the ray crosses the ally at (7,5) — floored


def test_support_call_computes_shot_or_approach():
    from phalanx.agent import _support_call

    base = {
        "q": 5,
        "r": 5,
        "gun_ready": True,
        "power_range": [3, 5, 7],
        "visible": [
            {"id": 4, "kind": "enemy", "dq": 3, "dr": 0, "dist": 3, "dir": 0, "clear_shot": True}
        ],
    }
    beacons = {"mate": "(9,5) t7 UNDER FIRE by #4 at (8,5)"}
    sc = _support_call(base, beacons, "me")
    assert sc["fire"] == 4 and "fire at it this turn" in sc["text"]  # attacker on my line: shoot

    blind = dict(base, visible=[])
    sc = _support_call(blind, beacons, "me")
    assert sc["move_to"] == (8, 5)  # attacker known but unseen: converge on it

    sc = _support_call(blind, {"mate": "(9,5) t7 UNDER FIRE by #4"}, "me")
    assert sc["move_to"] == (9, 5)  # attacker unknown: converge on the teammate

    assert _support_call(blind, {"mate": "(9,5) t7"}, "me") is None  # nobody under fire


def test_idle_tank_converges_on_ally_under_fire(monkeypatch):
    import asyncio

    from phalanx import agent as A

    p = {
        "id": 1,
        "q": 2,
        "r": 9,
        "heading": 0,
        "energy": 80,
        "gun_ready": True,
        "tick": 5,
        "width": 15,
        "height": 15,
        "map_radius": 7,
        "power_range": [3, 5, 7],
        "power_cost": [0, 2, 4],
        "fire_range": 7,
        "visible": [],
        "walls": [],
        "safe": True,
        "zone_radius": 14,
        "dist_center": 5,
        "to_center": 0,
        "hit_taken": 0,
        "last_fire": "",
    }

    async def idle_chat(http, system, transcript, force=None, toolset=None):
        return (
            "",
            [{"id": "c1", "name": "act", "input": {"move": "hold"}}],
            0,
            0,
            {"cin": 0.0, "cout": 0.0},
        )

    monkeypatch.setattr(A, "_chat", idle_chat)
    monkeypatch.setattr(A, "_consume_inbox", _noop_inbox)
    monkeypatch.setattr(A, "_board", _noop_board)
    monkeypatch.setattr(A, "_send", _noop_send)
    beacons = {"mate": "(9,5) t5 UNDER FIRE by #4 at (8,5)"}
    intent, cost, plan, inbox, recalled = asyncio.run(
        A.decide(None, {"id": "me", "key": "k"}, p, ["mate"], beacons=beacons, solo=False)
    )
    assert intent.get("move_to") == (8, 5) or intent.get("move") == "fwd"


async def _noop_inbox(http, agent):
    return "", {}


async def _noop_board(http, agent):
    return ""


async def _noop_send(http, agent, to, text, parents, subject=""):
    return None


def test_crossfire_floor_holds_the_shot_when_an_ally_hugs_the_line():
    from phalanx.agent import _sanitize_intent

    p = {
        "id": 1,
        "q": 5,
        "r": 5,
        "heading": 0,
        "energy": 60,
        "gun_ready": True,
        "power_range": [3, 5, 7],
        "power_cost": [0, 2, 4],
        "fire_range": 7,
        "visible": [
            {"id": 2, "kind": "ally", "dq": 2, "dr": -1, "dist": 2, "dir": 1},  # beside the line
            {
                "id": 5,
                "kind": "enemy",
                "dq": 4,
                "dr": 0,
                "dist": 4,
                "dir": 0,
                "clear_shot": True,
                "energy": 60,
            },
        ],
        "walls": [],
        "safe": True,
        "dist_center": 2,
        "to_center": 0,
    }
    intent = _sanitize_intent({"turn": 0, "move": "hold", "fire": 5, "power": 2}, p, {}, "me", True)
    assert not intent.get("fire")  # ally one step from the firing line, closer than the target


def test_burnout_floor_blocks_suicide_unless_finisher():
    from phalanx.agent import _sanitize_intent

    base = {
        "id": 1,
        "q": 5,
        "r": 5,
        "heading": 0,
        "energy": 2,
        "gun_ready": True,
        "power_range": [3, 5, 7],
        "power_cost": [0, 2, 4],
        "fire_range": 7,
        "visible": [
            {
                "id": 5,
                "kind": "enemy",
                "dq": 4,
                "dr": 0,
                "dist": 4,
                "dir": 0,
                "clear_shot": True,
                "energy": 60,
            }
        ],
        "walls": [],
        "safe": True,
        "dist_center": 2,
        "to_center": 0,
    }
    out = _sanitize_intent({"turn": 0, "move": "hold", "fire": 5, "power": 2}, base, {}, "me", True)
    assert not out.get("fire")  # power 2 costs 4 = death for nothing; power 1 cannot reach

    finisher = dict(base)
    finisher["visible"] = [dict(base["visible"][0], energy=10)]
    out = _sanitize_intent(
        {"turn": 0, "move": "hold", "fire": 5, "power": 2}, finisher, {}, "me", True
    )
    assert out.get("fire") == 5  # trading yourself for a kill stays a legal choice
