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


def test_repair_only_when_idle_untouched_and_inside():
    a, shooter, victim = _duel(41)
    victim.q, victim.r = 9, 5
    shooter.energy = 50.0
    shooter.cooldown = 1  # not firing this turn
    a.step()  # everyone idle
    assert shooter.energy == 52.0  # idle, untouched, inside: +2

    e0 = a.tanks[victim.id].energy
    from phalanx.tank import AXIAL_DIRS

    victim.heading = AXIAL_DIRS.index((1, 0))
    a.submit(victim.id, {"move": "fwd"})
    a.step()
    assert a.tanks[victim.id].energy == e0  # moved: no repair (and moving is free now)

    shooter.cooldown = 0
    shooter.energy = 50.0
    a.submit(shooter.id, {"fire": victim.id, "power": 3})
    a.step()
    # fired: cost 4, hit refund +2, and NO +2 repair on top
    assert a.tanks[shooter.id].energy == 48.0


def test_orders_focus_overrides_temperament_and_seeds_intel():
    from phalanx.control import Bot

    bot = Bot(1, "artel", "brawler")
    p = {
        "q": 5,
        "r": 5,
        "heading": 0,
        "energy": 80,
        "gun_ready": True,
        "visible": [
            {
                "id": 4,
                "kind": "enemy",
                "dq": 1,
                "dr": 0,
                "dist": 1,
                "dir": 0,
                "clear_shot": True,
                "energy": 60,
            },
            {
                "id": 5,
                "kind": "enemy",
                "dq": 3,
                "dr": 0,
                "dist": 3,
                "dir": 0,
                "clear_shot": True,
                "energy": 60,
            },
        ],
        "walls": [],
        "zone_radius": 14,
    }
    # no orders: a brawler shoots the nearest (#4)
    assert bot.decide(dict(p), DEFAULT, 1)["fire"] == 4
    # focus order: the called target outranks temperament
    bot.orders["focus"] = 5
    assert bot.decide(dict(p), DEFAULT, 2)["fire"] == 5
    # focus_at seeds the board with intel the tank never saw itself
    blind = dict(p, visible=[])
    bot2 = Bot(2, "artel", "ranger")
    bot2.orders.update({"focus": 5, "focus_at": (9, 5)})
    bot2.decide(blind, DEFAULT, 3)
    assert bot2.board[5]["q"] == 9 and bot2.board[5]["r"] == 5  # hunting a reported enemy


def test_orders_regroup_moves_and_clears_on_arrival():
    from phalanx.control import Bot

    bot = Bot(1, "artel", "opportunist")
    p = {
        "q": 3,
        "r": 10,
        "heading": 0,
        "energy": 80,
        "gun_ready": True,
        "visible": [],
        "walls": [],
        "zone_radius": 14,
    }
    bot.orders["regroup"] = (8, 6)
    out = bot.decide(dict(p), DEFAULT, 1)
    assert out["move"] == "fwd"  # marching to the rally, not sweeping its lane
    near = dict(p, q=8, r=7)  # adjacent to the rally point
    bot.decide(near, DEFAULT, 2)
    assert "regroup" not in bot.orders  # arrived: order clears itself


def test_orders_post_holds_position_when_nothing_known():
    from phalanx.control import Bot

    bot = Bot(1, "artel", "brawler")
    p = {
        "q": 7,
        "r": 7,
        "heading": 0,
        "energy": 80,
        "gun_ready": True,
        "visible": [],
        "walls": [],
        "zone_radius": 14,
    }
    bot.orders["post"] = (7, 7)
    out = bot.decide(dict(p), DEFAULT, 1)
    assert out["move"] == "hold"  # holding the assigned post instead of lane-sweeping


def test_red_bots_never_have_orders():
    # the ablation invariant: red runs the same Bot with orders forever empty
    from phalanx.control import Bot

    bot = Bot(4, "red")
    assert bot.orders == {}


def test_coordination_orders_beat_identical_solo_motors():
    # THE architecture's claim, regression-tested: same Bot motors on both sides; blue gets
    # a scripted commander with Artel-grade information (merged boards, focus calls, intel
    # feeds). Coordination must produce a real edge (~58% healthy; <46% means the orders
    # plumbing broke). The LLM commander only needs to approximate this policy.
    from collections import Counter

    from phalanx.control import STRATEGIES, Bot

    def command_blue(bots):
        board = {}
        for b in bots:
            for eid, rec in b.board.items():
                if eid not in board or rec["seen"] > board[eid]["seen"]:
                    board[eid] = rec
        if not board:
            return
        target = min(board, key=lambda e: (board[e]["energy"],))
        rec = board[target]
        for b in bots:
            b.orders["focus"] = target
            if target not in b.board:
                b.orders["focus_at"] = (rec["q"], rec["r"])

    wins = Counter()
    for m in range(120):
        a = Arena(DEFAULT, seed=9000 + m)
        a.seed_house(flip=bool(m % 2))
        blue, red = {}, {}
        for i, t in enumerate(a.tanks.values()):
            side = blue if t.team == "artel" else red
            side[t.id] = Bot(t.id, t.team, STRATEGIES[i % len(STRATEGIES)])
        for _ in range(DEFAULT.match_max_ticks + 1):
            command_blue([b for tid, b in blue.items() if tid in a.tanks])
            for t in list(a.tanks.values()):
                p = a.perceive(t.id)
                if p:
                    bot = blue.get(t.id) or red[t.id]
                    a.submit(t.id, bot.decide(p, a.cfg, a.tick_count))
            a.step()
            if a.winner:
                break
        wins[a.winner] += 1
    assert wins["artel"] >= 55, f"coordination edge lost: {dict(wins)}"


def test_hurt_bot_parks_to_repair_when_nothing_in_sight():
    from phalanx.control import Bot

    bot = Bot(1, "artel", "brawler")
    p = {
        "q": 7,
        "r": 7,
        "heading": 0,
        "energy": 20,
        "gun_ready": True,
        "visible": [],
        "walls": [],
        "zone_radius": 14,
        "safe": True,
    }
    out = bot.decide(dict(p), DEFAULT, 5)
    assert out["move"] == "hold"  # parked: banking +2/turn

    p["visible"] = [
        {
            "id": 5,
            "kind": "enemy",
            "dq": 2,
            "dr": 0,
            "dist": 2,
            "dir": 0,
            "clear_shot": True,
            "energy": 50,
        }
    ]
    out = bot.decide(dict(p), DEFAULT, 6)
    assert out["move"] != "hold" or out.get("fire")  # contact: fight or fall back, not nap


def test_recall_query_reflects_the_situation():
    from phalanx.agent import _recall_query

    class Engaged:
        board = {"e1": {"q": 1, "r": 1}}

    q = _recall_query({"hit_taken": True, "energy": 20, "safe": False}, Engaged(), {"victim": "t2"})
    assert "under fire" in q
    assert "ally under fire" in q
    assert "low energy" in q
    assert "zone" in q
    assert "focus fire" in q

    quiet = _recall_query({"energy": 100, "safe": True}, None, None)
    assert "hunting" in quiet
    assert "fire" not in quiet


def test_recalled_lessons_join_search_hits():
    import asyncio

    from phalanx.agent import _recall_lessons

    class Resp:
        status_code = 200

        def json(self):
            return [{"content": "[WIN] hold the corner"}, {"content": "[LOSS] strung out"}]

    class Http:
        async def get(self, url, headers=None, params=None):
            assert url.endswith("/memory/search")
            assert params["q"] and params["project"]
            return Resp()

    out = asyncio.run(_recall_lessons(Http(), {"id": "a", "key": "k"}, "under fire"))
    assert out == "[WIN] hold the corner | [LOSS] strung out"


def test_mutual_wipeout_is_a_draw_not_a_win():
    a = Arena(DEFAULT, seed=11)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    blue = next(t for t in a.tanks.values() if t.team == "artel")
    foe = next(t for t in a.tanks.values() if t.team == "red")
    a.walls.clear()
    for t in list(a.tanks.values()):  # only the duelists remain
        if t.id not in (blue.id, foe.id):
            del a.tanks[t.id]
    # red's desperation shot kills blue but burns red out (power 3 costs 4, refund only
    # gives 2 back) — both fall in the same step, the case seen live and scored as a win
    blue.q, blue.r, blue.cooldown, blue.energy = 5, 5, 0, 5
    foe.q, foe.r, foe.cooldown, foe.energy = 7, 5, 0, 2
    a.submit(foe.id, {"fire": blue.id, "power": 3})
    a.step()
    assert not a.tanks  # both fell this step
    assert a.winner is None
    assert a.draw
    assert a.stats()["draw"] is True


def test_match_stats_measure_the_lesson_families():
    a = Arena(DEFAULT, seed=12)
    a.add_team("artel", "house:artel", (5, 5))
    a.add_team("red", "house:red", (5, 5))
    blue = next(t for t in a.tanks.values() if t.team == "artel")
    foe = next(t for t in a.tanks.values() if t.team == "red")
    a.walls.clear()
    for i, o in enumerate(t for t in a.tanks.values() if t.id not in (blue.id, foe.id)):
        o.q, o.r = 1 + i, 12
    blue.q, blue.r, blue.cooldown, blue.energy = 5, 5, 0, 50
    foe.q, foe.r, foe.energy = 7, 5, 50
    a.submit(blue.id, {"fire": foe.id})
    a.step()
    s = a.match_stats["artel"]
    assert s["shots"] == 1 and s["shots_hit"] == 1 and s["trigger_energy"] > 0

    blue.cooldown = 0
    a.submit(blue.id, {"fire_at": [5, 9]})  # nothing on that line
    a.step()
    assert a.match_stats["artel"]["shots_missed"] == 1

    summary = a.stats_summary()
    assert "artel: 2 shots (1 hit, 1 missed" in summary
    assert "red: 0 shots" in summary  # red never fired but its repair economy still shows


def test_reflect_writes_nothing_when_the_lesson_is_already_known(monkeypatch):
    import asyncio

    from phalanx import agent as A

    seen = {}

    async def fake_oneshot(http, sys, user, *a, **k):
        seen["user"] = user
        return "NONE"

    monkeypatch.setattr(A, "_oneshot", fake_oneshot)
    out = asyncio.run(A._reflect(None, {}, "Your team WON.", "kill log", "[WIN] stick together"))
    assert out == ""
    assert "[WIN] stick together" in seen["user"]  # the corpus was shown to the model

    async def fake_novel(http, sys, user, *a, **k):
        return "When reloading, break line of sight."

    monkeypatch.setattr(A, "_oneshot", fake_novel)
    out = asyncio.run(A._reflect(None, {}, "Your team WON.", "kill log"))
    assert out == "When reloading, break line of sight."


def test_sdk_extract_call_parses_json_orders():
    from phalanx.agent import TOOLS
    from phalanx.sdkchat import extract_call, tool_instruction

    text, calls = extract_call('Sure! ```json\n{"focus": 4, "say": "FOCUS #4"}\n```', TOOLS)
    assert text == ""
    assert calls == [{"name": "command", "input": {"focus": 4, "say": "FOCUS #4"}}]

    text, calls = extract_call("no json here", TOOLS)
    assert calls == [] and text == "no json here"

    text, calls = extract_call("[1, 2]", TOOLS)  # JSON but not an object
    assert calls == []

    inst = tool_instruction(TOOLS)
    assert "'command'" in inst and '"focus"' in inst and "code fences" in inst
    assert tool_instruction(None) == "Reply with plain text only."


def test_chat_routes_claude_sdk_and_fails_over(monkeypatch):
    import asyncio

    from phalanx import agent as A

    sdk_ep = {"provider": "claude-sdk", "model": "haiku", "key": "tok", "cin": 0, "cout": 0}
    gem_ep = dict(A.PRIMARY, key="gk")
    monkeypatch.setattr(A, "_ENDPOINTS", [sdk_ep, gem_ep])
    monkeypatch.setattr(A, "_down_until", {})

    async def fake_sdk(ep, system, transcript, tools, session=""):
        return "", [{"name": "command", "input": {"focus": 7}}], 10, 5, dict(ep, flat_cost=0.004)

    monkeypatch.setattr(A, "sdk_chat", fake_sdk)
    text, calls, tin, tout, ep = asyncio.run(
        A._chat(None, "sys", [{"role": "user", "text": "brief"}], True, A.TOOLS)
    )
    assert calls[0]["input"]["focus"] == 7
    assert ep["flat_cost"] == 0.004  # SDK-reported credit cost wins over token math

    async def broken_sdk(ep, system, transcript, tools, session=""):
        raise RuntimeError("credit exhausted")

    posted = {}

    async def fake_post(url, headers=None, json=None):
        posted["url"] = url

        class R:
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

        return R()

    class H:
        post = staticmethod(fake_post)

    monkeypatch.setattr(A, "sdk_chat", broken_sdk)
    monkeypatch.setattr(A, "_down_until", {})
    asyncio.run(A._chat(H(), "sys", [{"role": "user", "text": "brief"}], True, A.TOOLS))
    assert posted["url"] == gem_ep["url"]  # failed SDK rolled to the Gemini endpoint
    assert A._down_until.get("haiku", 0) > 0  # and the SDK endpoint was benched


def test_spend_cap_is_monthly(monkeypatch):
    from phalanx import agent as A
    from phalanx.agent import SPEND_CAP_USD, Squad

    monkeypatch.setattr(A, "LLM_KEY", "test-key")  # the cap, not key presence, is under test
    sq = Squad(solo=True)
    sq.agents = [{"id": "x", "key": "k"}]
    sq.month, sq.month_spent = "2026-05", SPEND_CAP_USD + 1  # blew LAST month's budget
    assert sq.enabled  # a new month resets the meter
    assert sq.month_spent == 0.0
    sq.month_spent = SPEND_CAP_USD + 1  # blew THIS month's budget
    sq.month = (
        __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m")
    )
    assert not sq.enabled


def test_sdk_sessions_reuse_drop_and_reset(monkeypatch):
    import asyncio

    from phalanx import sdkchat as S

    class FakeResult:
        is_error = False
        result = '{"focus": 2}'
        usage = {"input_tokens": 10, "output_tokens": 5}
        total_cost_usd = 0.001

    class FakeClient:
        instances = []

        def __init__(self, opts):
            self.opts = opts
            self.queries = 0
            self.connected = False
            self.broken = False
            FakeClient.instances.append(self)

        async def connect(self):
            self.connected = True

        async def disconnect(self):
            self.connected = False

        async def query(self, prompt):
            if self.broken:
                raise RuntimeError("session died")
            self.queries += 1

        async def receive_response(self):
            from claude_agent_sdk import ResultMessage
            from unittest.mock import MagicMock

            msg = MagicMock(spec=ResultMessage)
            msg.is_error = False
            msg.result = '{"focus": 2}'
            msg.usage = {"input_tokens": 10, "output_tokens": 5}
            msg.total_cost_usd = 0.001
            yield msg

    import claude_agent_sdk as sdk

    monkeypatch.setattr(sdk, "ClaudeSDKClient", FakeClient)
    asyncio.run(S.reset_sessions())
    FakeClient.instances.clear()

    ep = {"provider": "claude-sdk", "model": "haiku"}
    from phalanx.agent import TOOLS

    async def run():
        # two commands on the same session: ONE client, spawn paid once
        await S.sdk_chat(ep, "sys", [{"text": "brief 1"}], TOOLS, session="blue-1")
        await S.sdk_chat(ep, "sys", [{"text": "brief 2"}], TOOLS, session="blue-1")
        assert len(FakeClient.instances) == 1
        assert FakeClient.instances[0].queries == 2

        # a broken session is dropped, never reused
        FakeClient.instances[0].broken = True
        try:
            await S.sdk_chat(ep, "sys", [{"text": "brief 3"}], TOOLS, session="blue-1")
        except RuntimeError:
            pass
        assert "blue-1" not in S._sessions

        # match reset tears everything down
        await S.sdk_chat(ep, "sys", [{"text": "brief 4"}], TOOLS, session="blue-2")
        assert S._sessions
        await S.reset_sessions()
        assert not S._sessions
        assert all(not c.connected for c in FakeClient.instances)

    asyncio.run(run())


def test_comms_from_emits_radio_events_with_intel_cells():
    from phalanx.agent import _comms_from

    class B:
        id = 2
        orders = {"focus": 5, "focus_at": (7, 4)}

    p = {"tick": 12}
    out = _comms_from({"say": "SPOTTED #5 (7,4)", "focus": 5, "focus_at": [7, 4]}, B(), p)
    kinds = {e["kind"]: e for e in out}
    assert "say" in kinds and kinds["say"]["text"] == "SPOTTED #5 (7,4)"
    # focus_at present -> a VECTOR intel event carrying the hunted cell (the ping)
    assert "intel" in kinds and kinds["intel"]["cell"] == [7, 4]
    assert all(e["tank"] == 2 and e["t"] == 12 for e in out)

    # plain focus (no focus_at) is a focus event, no cell
    class B2:
        id = 3
        orders = {"focus": 9}

    out2 = _comms_from({"focus": 9, "regroup": [8, 6]}, B2(), {"tick": 3})
    k2 = {e["kind"]: e for e in out2}
    assert k2["focus"]["text"] == "FOCUS #9" and "cell" not in k2["focus"]
    assert k2["rally"]["cell"] == [8, 6]

    # empty command this call -> nothing on the feed, even if the bot holds prior orders
    assert _comms_from({}, B2(), {"tick": 1}) == []
