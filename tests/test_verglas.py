import asyncio

import verglas.engine as E
from verglas.brain import make_decider
from verglas.engine import Meeting, new_game
from verglas.meeting import run_canned_meeting


def _run(coro):
    return asyncio.run(coro)


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
