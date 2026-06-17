from __future__ import annotations

import statistics

from pitch.dist import DistTeam, make_dist_brain
from pitch.engine import Pitch, _len

HERD_NEAR = 7.0
MARK_NEAR = 6.0


def _instrument(p: Pitch, m: dict) -> None:
    c = p.cfg
    b = p.ball
    poss_team = p.players[p.possessor].team if p.possessor is not None else None
    if poss_team:
        m["poss"][poss_team] += 1
    for team in ("home", "away"):
        mates = [pl for pl in p.players if pl.team == team and pl.role != "GK"]
        opps = [pl for pl in p.players if pl.team != team]
        # thundering herd: 3+ of our outfielders swarming the ball when we don't have it
        if poss_team != team:
            near = sum(1 for pl in mates if _len(pl.x - b.x, pl.y - b.y) < HERD_NEAR)
            if near >= 3:
                m["herd"][team] += 1
        # mutual-exclusion violation: two of our players marking the same opponent (wasted lock)
        for o in opps:
            coverers = sum(1 for pl in mates if _len(pl.x - o.x, pl.y - o.y) < MARK_NEAR)
            if coverers >= 2:
                m["double"][team] += 1
    # dropped request: a possessor in shooting range with no defender near -> the OTHER team conceded it
    if poss_team:
        gx, gy = p.attack_goal(poss_team)
        shooter = p.players[p.possessor]
        if _len(gx - shooter.x, gy - shooter.y) < c.shoot_range:
            defenders = [pl for pl in p.players if pl.team != poss_team and pl.role != "GK"]
            if not any(_len(pl.x - shooter.x, pl.y - shooter.y) < MARK_NEAR for pl in defenders):
                conceder = "away" if poss_team == "home" else "home"
                m["freeshot"][conceder] += 1


def run(seed: int, home_bus: bool, away_bus: bool) -> tuple:
    p = Pitch(seed=seed)
    p.setup(["x"] * 9, ["y"] * 9)
    home, away = DistTeam("home", home_bus), DistTeam("away", away_bus)
    brain = make_dist_brain(home, away)
    m = {k: {"home": 0, "away": 0} for k in ("poss", "herd", "double", "freeshot", "patt", "pcomp")}
    # count passes attempted/completed per team (credibility: a blind team should still pass, not flail)
    orig = p._begin_pass

    def patched(owner, rid):
        m["patt"][owner.team] += 1
        ic = p._lane_interceptor(owner, p.players[rid])
        orig(owner, rid)
        if ic is None and p.restart_kind != "offside":
            m["pcomp"][owner.team] += 1

    p._begin_pass = patched
    while p.tick < p.cfg.match_ticks:
        p.step(brain)
        _instrument(p, m)
    return p.score, m


def main() -> None:
    N = 40
    # tally everything from the BUS-ON team's perspective and the BUS-OFF team's perspective,
    # alternating which side holds the bus so home advantage cancels.
    agg = {
        on: {k: [] for k in ("gf", "ga", "poss", "herd", "double", "freeshot", "patt", "pcomp")}
        for on in ("bus_on", "bus_off")
    }
    wins = {"bus_on": 0, "bus_off": 0, "draw": 0}
    for s in range(N):
        home_bus = s % 2 == 0  # alternate
        score, m = run(s, home_bus, not home_bus)
        on_side = "home" if home_bus else "away"
        off_side = "away" if home_bus else "home"
        total_poss = m["poss"]["home"] + m["poss"]["away"] or 1
        for side, key in ((on_side, "bus_on"), (off_side, "bus_off")):
            other = "away" if side == "home" else "home"
            agg[key]["gf"].append(score[side])
            agg[key]["ga"].append(score[other])
            agg[key]["poss"].append(m["poss"][side] / total_poss * 100)
            agg[key]["herd"].append(m["herd"][side])
            agg[key]["double"].append(m["double"][side])
            agg[key]["freeshot"].append(m["freeshot"][side])
            agg[key]["patt"].append(m["patt"][side])
            agg[key]["pcomp"].append(m["pcomp"][side])
        if score[on_side] > score[off_side]:
            wins["bus_on"] += 1
        elif score[off_side] > score[on_side]:
            wins["bus_off"] += 1
        else:
            wins["draw"] += 1

    def mean(x):
        return round(statistics.mean(x), 1)

    print(f"=== {N} matches, symmetric nodes, only difference = the bus ===\n")
    print(f"{'metric':<28}{'BUS ON':>10}{'BUS OFF':>10}")
    print(f"{'-' * 48}")
    print(f"{'goals scored':<28}{mean(agg['bus_on']['gf']):>10}{mean(agg['bus_off']['gf']):>10}")
    print(f"{'goals conceded':<28}{mean(agg['bus_on']['ga']):>10}{mean(agg['bus_off']['ga']):>10}")
    print(
        f"{'possession %':<28}{mean(agg['bus_on']['poss']):>10}{mean(agg['bus_off']['poss']):>10}"
    )
    print(
        f"{'herd ticks (swarm ball)':<28}{mean(agg['bus_on']['herd']):>10}{mean(agg['bus_off']['herd']):>10}"
    )
    print(
        f"{'double-mark ticks':<28}{mean(agg['bus_on']['double']):>10}{mean(agg['bus_off']['double']):>10}"
    )
    print(
        f"{'free-shots conceded':<28}{mean(agg['bus_on']['freeshot']):>10}{mean(agg['bus_off']['freeshot']):>10}"
    )
    print(
        f"{'passes attempted':<28}{mean(agg['bus_on']['patt']):>10}{mean(agg['bus_off']['patt']):>10}"
    )

    def comp(k):
        a, c2 = sum(agg[k]["patt"]), sum(agg[k]["pcomp"])
        return round(c2 / a * 100) if a else 0

    print(f"{'pass completion %':<28}{comp('bus_on'):>10}{comp('bus_off'):>10}")
    print(f"\nrecord: BUS ON {wins['bus_on']}W  BUS OFF {wins['bus_off']}W  {wins['draw']}D")


if __name__ == "__main__":
    main()
