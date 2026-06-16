from __future__ import annotations

from .bot import decide
from .config import DEFAULT
from .engine import Pitch, _len

HOME = ["H-GK", "H-DEF1", "H-DEF2", "H-MID", "H-FWD"]
AWAY = ["A-GK", "A-DEF1", "A-DEF2", "A-MID", "A-FWD"]


def _ascii(pitch: Pitch, cols: int = 62, rows: int = 18) -> str:
    c = pitch.cfg
    grid = [[" "] * cols for _ in range(rows)]

    def cell(x: float, y: float) -> tuple[int, int]:
        cx = min(cols - 1, max(0, int(x / c.length * (cols - 1))))
        ry = min(rows - 1, max(0, int(y / c.width * (rows - 1))))
        return ry, cx

    for r in range(rows):
        grid[r][0] = grid[r][cols - 1] = "|"
    for p in pitch.players:
        r, cc = cell(p.x, p.y)
        grid[r][cc] = "H" if p.team == "home" else "a"
    r, cc = cell(pitch.ball.x, pitch.ball.y)
    grid[r][cc] = "O"
    return "\n".join("".join(row) for row in grid)


def run(ticks: int = DEFAULT.match_ticks, seed: int = 7, show: bool = True) -> dict:
    pitch = Pitch(seed=seed)
    pitch.setup(HOME, AWAY)
    poss = {"home": 0, "away": 0, "loose": 0}
    swarm_total = 0.0
    near_ball_max = 0
    ball_x_min, ball_x_max = 1e9, -1e9
    snaps_at = {int(ticks * f) for f in (0.25, 0.5, 0.75)}

    for t in range(ticks):
        pitch.step(decide)
        if pitch.possessor is not None:
            poss[pitch.players[pitch.possessor].team] += 1
        else:
            poss["loose"] += 1
        near = sum(1 for p in pitch.players if _len(p.x - pitch.ball.x, p.y - pitch.ball.y) < 9.0)
        swarm_total += near
        near_ball_max = max(near_ball_max, near)
        ball_x_min = min(ball_x_min, pitch.ball.x)
        ball_x_max = max(ball_x_max, pitch.ball.x)
        if show and t in snaps_at:
            print(f"\n--- tick {t}  score {pitch.score['home']}-{pitch.score['away']} ---")
            print(_ascii(pitch))

    total_poss = poss["home"] + poss["away"] or 1
    out = {
        "score": dict(pitch.score),
        "goals": pitch.score["home"] + pitch.score["away"],
        "possession_home_pct": round(100 * poss["home"] / total_poss, 1),
        "loose_pct": round(100 * poss["loose"] / ticks, 1),
        "avg_players_near_ball": round(swarm_total / ticks, 2),
        "max_players_near_ball": near_ball_max,
        "ball_x_span": round(ball_x_max - ball_x_min, 1),
        "field_length": pitch.cfg.length,
        "events": pitch.events,
    }
    return out


if __name__ == "__main__":
    res = run()
    print("\n=== MATCH SUMMARY ===")
    for k, v in res.items():
        if k != "events":
            print(f"  {k}: {v}")
    print("  goal log:")
    for e in res["events"]:
        print(f"    {e}")
    print(
        "\n  read: goals>0 = it scores; avg_near_ball ~1-4 (not ~10) = NOT a swarm; "
        "ball_x_span near field_length = play traverses the pitch."
    )
