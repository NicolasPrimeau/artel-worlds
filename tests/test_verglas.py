import asyncio
import random
from collections import deque

import verglas.engine as E
from verglas.brain import make_decider
from verglas.engine import (
    MIN_ROOMS,
    Meeting,
    _generate_station,
    _room_count,
    new_game,
)
from verglas.meeting import run_canned_meeting


def _run(coro):
    return asyncio.run(coro)


def _reachable_rooms(rects, corr):
    # flood the WALKABLE set (room interiors + corridor tiles) the renderer's path router uses,
    # and report which rooms are physically reachable from an arbitrary starting room
    walk = set(map(tuple, corr))
    tile_room = {}
    for n, (x, y, w, h) in rects.items():
        for tx in range(x, x + w):
            for ty in range(y, y + h):
                walk.add((tx, ty))
                tile_room[(tx, ty)] = n
    start = next(iter(rects))
    sx, sy, sw, sh = rects[start]
    src = (sx + sw // 2, sy + sh // 2)
    seen, q, hit = {src}, deque([src]), {start}
    while q:
        x, y = q.popleft()
        if (x, y) in tile_room:
            hit.add(tile_room[(x, y)])
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (x + dx, y + dy)
            if nb in walk and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return hit


def test_every_room_is_reachable_in_every_generated_station():
    # one connected walkable network: every room must be reachable over the walkable-tile set, or the
    # path router can't route into it and agents clip through the wall. The live server seeds randomly
    # across [1, 2**31), so sample THAT range (sequential seeds never exposed the cluster-split bug).
    pick = random.Random(1234)
    for _ in range(800):
        seed = pick.randint(1, 2**31 - 1)
        names, adj, vents, rects, doors, centers, corr = _generate_station(random.Random(seed))
        hit = _reachable_rooms(rects, corr)
        assert hit == set(names), (seed, set(names) - hit)


def test_room_count_scales_with_the_crew():
    assert _room_count(4) == MIN_ROOMS  # clamped up — never too few to play
    assert _room_count(100) == 12  # capped at the named-room list
    assert _room_count(8) < _room_count(12)  # fewer agents → fewer rooms


def test_scaled_stations_are_reachable_and_correctly_sized():
    # the floorplan now scales to a target room count; every size must still be one connected
    # walkable network with exactly that many named rooms, or the path router clips through walls
    pick = random.Random(99)
    for nr in range(MIN_ROOMS, 13):
        for _ in range(60):
            seed = pick.randint(1, 2**31 - 1)
            names, adj, vents, rects, doors, centers, corr = _generate_station(
                random.Random(seed), nr
            )
            assert len(names) == nr, (nr, len(names), seed)
            hit = _reachable_rooms(rects, corr)
            assert hit == set(names), (nr, seed, set(names) - hit)


def test_dark_knobs_scale_with_station_size():
    # START_DARK / DARK_CAP / INTEGRITY_FREE_DARK are tuned at 12 rooms; on a smaller station they must
    # scale or it's unplayable (every room dark at dawn, or the storm can never be capped under the count)
    for n in (6, 8, 10, 14):
        g = new_game(1, n, 1)
        nr = len(g.rooms)
        assert g.dark_cap <= nr  # the storm can never demand more dark rooms than exist
        assert len(g.dark) <= g.dark_cap  # don't open the night already past the cap
        assert g.free_dark < g.dark_cap  # always some drain pressure before a blackout is reachable


def test_room_adjacency_graph_is_connected():
    for seed in range(300):
        names, adj, vents, rects, doors, centers, corr = _generate_station(random.Random(seed))
        seen, stack = {names[0]}, [names[0]]
        while stack:
            n = stack.pop()
            for m in adj[n]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        assert seen == set(names), (seed, set(names) - seen)


def test_cold_alone_with_its_kill_does_not_report_only_crew_does():
    g = new_game(5, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    victim = next(a for a in g.living() if not a.impostor)
    room = "Mess Hall"
    cold.room = victim.room = room
    victim.alive = False
    g.bodies[room] = victim.id
    for a in g.living(impostor=False):
        a.room = (
            "Reactor"  # every living crewmate is elsewhere — only the Cold stands over the body
        )

    assert g._report_body() is None  # the Cold alone over the body it made never triggers a meeting
    assert room in g.bodies  # the corpse waits in the dark

    crew = next(a for a in g.living(impostor=False))
    crew.room = room
    mt = g._report_body()
    assert mt is not None and mt.reporter == crew.id  # a crewmate walking in opens it


def test_cold_leaves_the_room_immediately_after_a_kill():
    g = new_game(5, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    victim = next(a for a in g.living() if not a.impostor)
    room = next(r for r in g.rooms if g.adj.get(r))
    for a in g.living():
        a.room = "__elsewhere__"  # clear the room so the strike is unwitnessed
    cold.room = victim.room = room
    cold.gx, cold.gy = victim.gx, victim.gy = 5.0, 5.0  # in reach
    g.dark.add(room)
    g.cd = 0
    g.tick = 999

    assert g.do_kill(cold, victim.id) is True
    assert g.bodies.get(room) == victim.id  # the body stays where it fell
    assert cold.room != room  # ...but the Cold is already gone


def test_cold_flees_toward_the_dark_after_a_kill():
    g = new_game(7, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    room = next(r for r in g.rooms if len(g.adj.get(r, ())) >= 2)
    nbrs = sorted(g.adj[room])
    for a in g.living():
        a.room = "__elsewhere__"
    cold.room = room
    g.vents = {}  # force the on-foot path so the choice is among the neighbours
    g.dark = {nbrs[0]}  # exactly one neighbour is dark — the Cold should pick it

    g._flee_body(cold)
    assert cold.room == nbrs[0]  # retreats into the dark, not a lit neighbour


def test_a_killed_crewmate_frees_its_task():
    g = new_game(5, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    victim = next(a for a in g.living() if not a.impostor)
    room = next(r for r in g.rooms if g.adj.get(r))
    for a in g.living():
        a.room = "__elsewhere__"
    cold.room = victim.room = room
    cold.gx, cold.gy = victim.gx, victim.gy = 5.0, 5.0
    g.dark.add(room)
    g.cd = 0
    g.tick = 999
    task_room = next(r for r in g.rooms if r != room)
    if task_room in g.open_tasks:
        g.open_tasks.remove(task_room)
    victim.dest = task_room  # the victim had claimed a relight in another room

    assert g.do_kill(cold, victim.id) is True
    assert task_room in g.open_tasks  # the claim is freed back to the board for the living
    assert victim.dest is None and victim.tasking is False


def test_cold_can_kill_in_a_lit_room_only_when_the_victim_is_alone():
    g = new_game(5, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    crew = [a for a in g.living() if not a.impostor]
    victim = crew[0]
    room = next(r for r in g.rooms if g.adj.get(r) and r not in g.dark)
    for a in g.living():
        a.room = "__elsewhere__"
    cold.room = victim.room = room  # a LIT room (not in g.dark)
    cold.gx, cold.gy = victim.gx, victim.gy = 5.0, 5.0
    g.cd, g.tick = 0, 999
    assert room not in g.dark
    assert g.do_kill(cold, victim.id) is True  # alone with it in the light → killable now

    # but a second crewmate in the lit room sees everything — no kill
    g2 = new_game(6, 8, 2)
    cold2 = next(a for a in g2.living() if a.impostor)
    c2 = [a for a in g2.living() if not a.impostor]
    v2, w2 = c2[0], c2[1]
    room2 = next(r for r in g2.rooms if g2.adj.get(r) and r not in g2.dark)
    for a in g2.living():
        a.room = "__elsewhere__"
    cold2.room = v2.room = w2.room = room2
    cold2.gx, cold2.gy = v2.gx, v2.gy = w2.gx, w2.gy = 5.0, 5.0
    g2.cd, g2.tick = 0, 999
    assert room2 not in g2.dark
    assert g2.do_kill(cold2, v2.id) is False  # a witness in the lit room blocks it


def test_no_kill_with_a_bystander_in_the_room_even_in_the_dark():
    g = new_game(7, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    crew = [a for a in g.living() if not a.impostor]
    victim, bystander = crew[0], crew[1]
    room = next(r for r in g.rooms if g.adj.get(r))
    for a in g.living():
        a.room = "__elsewhere__"
    cold.room = victim.room = bystander.room = room
    cold.gx, cold.gy = victim.gx, victim.gy = 5.0, 5.0
    bystander.gx, bystander.gy = 60.0, 60.0  # way across the room — but still IN it
    g.dark.add(room)  # and it's pitch dark
    g.cd, g.tick = 0, 999

    assert (
        g.do_kill(cold, victim.id) is False
    )  # any other crew in the room blocks it — distance/dark irrelevant


def test_a_second_cold_doesnt_block_a_kill_but_a_crewmate_does():
    g = new_game(3, 8, 2)
    colds = [a for a in g.living() if a.impostor]
    victim = next(a for a in g.living() if not a.impostor)
    room = next(r for r in g.rooms if g.adj.get(r))
    for a in g.living():
        a.room = "__elsewhere__"
    colds[0].room = colds[1].room = victim.room = room  # killer + an ALLY Cold + the victim
    colds[0].gx, colds[0].gy = victim.gx, victim.gy = 5.0, 5.0
    g.dark.add(room)
    g.cd, g.tick = 0, 999
    assert g.do_kill(colds[0], victim.id) is True  # an ally Cold isn't a witness → the kill stands

    # but any CREWMATE in the room blocks it
    g2 = new_game(4, 8, 2)
    cold2 = next(a for a in g2.living() if a.impostor)
    crew2 = [a for a in g2.living() if not a.impostor]
    v2, bystander = crew2[0], crew2[1]
    room2 = next(r for r in g2.rooms if g2.adj.get(r))
    for a in g2.living():
        a.room = "__elsewhere__"
    cold2.room = v2.room = bystander.room = room2
    cold2.gx, cold2.gy = v2.gx, v2.gy = 5.0, 5.0
    g2.cd, g2.tick = 0, 999
    assert g2.do_kill(cold2, v2.id) is False  # a crewmate witnesses → no kill


def test_cold_steers_clear_of_a_body_room():
    g = new_game(5, 8, 2)
    cold = next(a for a in g.living() if a.impostor)
    room = next(r for r in g.rooms if g.adj.get(r))
    g.bodies[room] = 999  # a corpse lies here
    cold.room = room
    cold.dest = room  # ...and the Cold is standing in it / headed back to it

    g._avoid_bodies()
    assert cold.room != room  # it slipped out before anyone could walk in
    assert cold.dest is None  # and dropped the intent to return


def test_body_finder_opens_the_meeting_with_context():
    g = new_game(5, 8, 2)
    crew = [a for a in g.living() if not a.impostor]
    finder, victim = crew[0], crew[1]
    victim.alive = False
    finder.found.append((g.tick, victim.room, victim.id))
    mt = Meeting(g.tick, finder.id, victim.room, victim.id)

    _run(run_canned_meeting(g, mt, make_decider(share=True)))

    assert mt.transcript[0][0] == finder.id  # the reporter speaks first
    opening = mt.transcript[0][1]
    assert victim.name in opening and victim.room in opening  # ...and gives the who + where


def test_meetings_only_happen_on_a_body():
    # no emergency button: a meeting only ever opens as a body report, so every meeting carries a victim
    for seed in range(40):
        g = new_game(seed, 8, 2)
        for _ in range(E.MAX_TICKS):
            mt = g.step()
            if mt is not None:
                assert mt.victim is not None, seed
                break
            if g.winner is not None:
                break


def test_no_parity_shortcut():
    # two Cold vs two crew used to be an instant impostor win; now it plays on
    g = new_game(3, 8, 2)
    crew = [a for a in g.agents if not a.impostor]
    for a in crew[2:]:
        a.alive = False
    g._check_win()
    assert g.winner is None


def test_cold_wins_only_by_taking_the_last_crewmate():
    g = new_game(3, 8, 2)
    crew = [a for a in g.agents if not a.impostor]
    for a in crew[1:]:
        a.alive = False  # one crew left, the Cold still standing
    g._check_win()
    assert g.winner is None  # not a win yet — no parity shortcut
    for _ in range(12):
        g.step()
        if g.winner:
            break
    assert g.winner == "impostor" and g.win_by == "extinction"
    assert not [a for a in g.agents if a.alive and not a.impostor]  # the last crew is gone


def test_final_hunt_flags_and_runs_down_the_survivor():
    g = new_game(5, 8, 2)
    crew = [a for a in g.agents if not a.impostor]
    for a in crew[1:]:
        a.alive = False
    g.step()
    assert g.hunting is True  # the mask is off
    # the hunt can't last forever — within a few ticks the survivor is run down
    for _ in range(8):
        g.step()
        if g.winner:
            break
    assert g.winner == "impostor"


def test_crew_still_win_by_ejecting_the_cold():
    g = new_game(3, 8, 2)
    for a in g.agents:
        if a.impostor:
            a.alive = False
    g._check_win()
    assert g.winner == "crew" and g.win_by == "ejection"


def test_fame_leaderboard_tally_and_endpoint():
    from starlette.testclient import TestClient

    import verglas.server as S

    S.G.fame = {}
    S.G.g.winner = "impostor"  # the Cold won this game
    S.G.record_result()
    imp = next(a for a in S.G.g.agents if a.impostor)
    crew = next(a for a in S.G.g.agents if not a.impostor)
    assert S.G.fame[imp.name]["cold"] == 1 and S.G.fame[imp.name]["coldWins"] == 1
    assert S.G.fame[crew.name]["crewWins"] == 0  # crew lost
    r = TestClient(S.app).get("/fame.json")
    assert r.status_code == 200 and "rows" in r.json()
