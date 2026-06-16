from __future__ import annotations

from dataclasses import dataclass

from . import bot
from .engine import Pitch, Player, _clamp

# The Artel team's coach. The baseline brain is good but STATIC — it plays the same way at 0-0 and
# 1-0 with five minutes left, and it never targets a specific opponent's weakness. The coach reads
# the live game every window and re-throws the plan: attack the opponent's weakest channel, and
# commit or hold numbers by the scoreline and clock. That adaptation is the edge a fixed team can't
# answer. (Step 1 computes the plan deterministically; step 2 routes it + per-player marking claims
# through the real Artel server; step 3 swaps this deterministic coach for an LLM that reads more.)


@dataclass
class Plan:
    overload_y: float  # the channel to attack — the opponent's weakest defensive side
    commit: int  # how many midfielders push up to join the attack (chasing the game)
    low_block: bool  # sit deeper and protect (seeing out a lead)


def optimize_lineup(players: list[Player]) -> None:
    # the coach's team-sheet: fit the rolled attribute-sets to roles — best handler in goal,
    # strength at the back, pace + finishing up top, control in midfield.
    pool = [(p.pace, p.acc, p.finishing, p.control, p.strength, p.handling) for p in players]
    used = [False] * len(pool)
    fit = {
        "GK": lambda s: s[5],
        "FWD": lambda s: s[2] + s[0],
        "DEF": lambda s: s[4] + s[0] * 0.4,
        "MID": lambda s: s[3] + s[1],
    }
    for p in sorted(players, key=lambda q: {"GK": 0, "FWD": 1, "DEF": 2, "MID": 3}.get(q.role, 4)):
        scorer = fit.get(p.role, sum)
        bi = max((i for i in range(len(pool)) if not used[i]), key=lambda i: scorer(pool[i]))
        used[bi] = True
        p.pace, p.acc, p.finishing, p.control, p.strength, p.handling = pool[bi]


def plan_for(pitch: Pitch, team: str) -> Plan:
    c = pitch.cfg
    other = "away" if team == "home" else "home"
    foe_def = [o for o in pitch.players if o.team == other and o.role in ("DEF", "MID")]

    # ATTACK THE WEAK CHANNEL: the opponent's flank with the least defensive resistance — fewest
    # bodies, and weakest (slow/weak) defenders. The static baseline never reads this.
    def side_strength(lo: float, hi: float) -> float:
        return sum(o.strength + o.pace for o in foe_def if lo <= o.y < hi) + 0.01 * (hi - lo)

    top = side_strength(0, c.width / 2)
    bot_ = side_strength(c.width / 2, c.width)
    overload_y = c.width * 0.27 if top <= bot_ else c.width * 0.73

    # SCORELINE + CLOCK: chase a deficit (commit numbers, push), protect a late lead (low block).
    gd = pitch.score[team] - pitch.score[other]
    frac_left = max(0.0, (c.match_ticks - pitch.tick) / c.match_ticks)
    late = frac_left < 0.45
    commit = 2 if (gd < 0 and late) else (1 if gd < 0 else 0)
    low_block = gd > 0 and late
    return Plan(overload_y=overload_y, commit=commit, low_block=low_block)


def coordinated_decide(pitch: Pitch, p: Player, plan: Plan) -> dict:
    # Defense stays exactly the baseline (it's already strong; overriding it only hurt). The edge is
    # adaptive ATTACK: funnel our play into the opponent's weak channel, and push extra runners
    # forward when chasing — or hold shape when protecting a lead.
    c = pitch.cfg
    teammate_ids = {q.id for q in pitch.teammates(p)}
    we_have_it = pitch.possessor is not None and pitch.possessor in teammate_ids

    if pitch.possessor == p.id:
        intent = bot.decide(pitch, p)
        if "kick" not in intent:  # carrying — drive at the weak channel, not just the open lane
            gx, _ = pitch.attack_goal(p.team)
            intent = {"move": (gx, _clamp(p.y * 0.4 + plan.overload_y * 0.6, 8, c.width - 8))}
        return intent

    if we_have_it and not plan.low_block:
        # support: the most-advanced `commit` midfielders break forward into the weak channel to
        # make the overload; everyone else holds the baseline shape.
        if p.role == "MID" and plan.commit > 0:
            mids = sorted(
                (q for q in pitch.teammates(p) if q.role == "MID"),
                key=lambda q: bot._fwd_x(q.team, q.x, c.length),
                reverse=True,
            )
            if p.id in {q.id for q in mids[: plan.commit]}:
                gx, _ = pitch.attack_goal(p.team)
                push = gx - (22 if p.team == "home" else -22)
                return {"move": (push, _clamp(plan.overload_y, 8, c.width - 8)), "kick": None}
        if p.role == "FWD":  # forwards lead the line in the weak channel
            base = bot._formation_target(pitch, p)
            return {"move": (base[0], _clamp(plan.overload_y, 8, c.width - 8)), "kick": None}

    return bot.decide(pitch, p)  # defend / press / hold — all baseline


def make_brain(artel_team: str | None):
    def brain(pitch: Pitch, p: Player) -> dict:
        if artel_team is not None and p.team == artel_team:
            plan = getattr(pitch, "_plan", None) or plan_for(pitch, artel_team)
            return coordinated_decide(pitch, p, plan)
        return bot.decide(pitch, p)

    return brain
