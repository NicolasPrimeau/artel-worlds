from pitch.bot import decide
from pitch.engine import Pitch, _len

HOME = ["GK a", "DEF b", "DEF c", "MID d", "FWD e"]
AWAY = ["GK f", "DEF g", "DEF h", "MID i", "FWD j"]


def _play(seed: int, ticks: int = 900):
    p = Pitch(seed=seed)
    p.setup(HOME, AWAY)
    near_sum = 0.0
    for _ in range(ticks):
        p.step(decide)
        near_sum += sum(1 for q in p.players if _len(q.x - p.ball.x, q.y - p.ball.y) < 9.0)
    return p, near_sum / ticks


def test_match_does_not_swarm_the_ball():
    # the one rule that makes it read like soccer: only the nearest pursues, the rest hold shape.
    _p, avg_near = _play(7)
    assert avg_near < 5.0, f"too many players clustering on the ball ({avg_near}) — that's a swarm"


def test_play_traverses_the_whole_pitch():
    p = Pitch(seed=3)
    p.setup(HOME, AWAY)
    xmin, xmax = 1e9, -1e9
    for _ in range(900):
        p.step(decide)
        xmin, xmax = min(xmin, p.ball.x), max(xmax, p.ball.x)
    assert (xmax - xmin) > p.cfg.length * 0.6  # the ball is not stuck in one zone


def test_is_deterministic():
    a, _ = _play(11)
    b, _ = _play(11)
    assert a.score == b.score  # same seed -> identical match (paired/replayable like phalanx)


def test_snapshot_is_well_formed():
    from pitch.server import Game

    g = Game()
    g.pitch.step(decide)
    s = g.snapshot()
    assert {"home", "away", "ball", "players", "tick", "match_ticks"} <= set(s)
    assert len(s["players"]) == 10
    assert s["home"]["club"] and s["away"]["club"]
