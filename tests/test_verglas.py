import asyncio
import random
from collections import deque

import verglas.engine as E
from verglas.brain import make_decider
from verglas.engine import Meeting, _generate_station, new_game
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
    # no sealed rooms: every room must connect to the corridor network through a doorway, or the path
    # router can't route into it and agents clip through the wall. Checked across many seeds.
    for seed in range(300):
        names, adj, vents, rects, doors, centers, corr = _generate_station(random.Random(seed))
        hit = _reachable_rooms(rects, corr)
        assert hit == set(names), (seed, set(names) - hit)


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


def test_emergency_caller_opens_the_meeting():
    g = new_game(7, 8, 2)
    caller = next(a for a in g.living() if not a.impostor)
    mt = Meeting(g.tick, caller.id, E.HUB, None)

    _run(run_canned_meeting(g, mt, make_decider(share=True)))

    assert mt.transcript[0][0] == caller.id


def test_impostor_can_call_an_emergency_meeting(monkeypatch):
    g = new_game(5, 8, 2)
    monkeypatch.setattr(E, "EMERGENCY_P", 0.0)
    monkeypatch.setattr(E, "IMPOSTOR_EMERGENCY_P", 1.0)
    g.cd = 4  # after this tick's decrement -> 3: no kill (needs 0) and past the post-kill delay
    mt = g.step()
    assert mt is not None
    assert g.by_id(mt.reporter).impostor
    assert mt.victim is None


def test_impostor_will_not_call_one_right_after_a_kill(monkeypatch):
    g = new_game(5, 8, 2)
    monkeypatch.setattr(E, "EMERGENCY_P", 0.0)
    monkeypatch.setattr(E, "IMPOSTOR_EMERGENCY_P", 1.0)
    g.cd = E.KILL_CD  # just killed -> still inside the post-kill delay window
    assert g.step() is None
