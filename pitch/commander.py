from __future__ import annotations

import os
import time
from dataclasses import dataclass

from . import artel_client, bot, llm
from .engine import Pitch, Player, _clamp

LLM_EVERY = float(os.environ.get("PITCH_LLM_EVERY", "6"))  # seconds between coach LLM calls

COACH_SYSTEM = (
    "You are the head coach of an AI soccer team in a live 2D match. Read the situation and set the "
    "game plan. Reply with ONLY a JSON object, no prose: "
    '{"overload":"left|right|center","commit":0-3,"low_block":true|false}. '
    "overload = which flank to attack; commit = how many midfielders to push forward (more when "
    "chasing a deficit); low_block = sit deep to protect a lead. Be decisive and adapt to the score, "
    "the clock, and the opponent's weak side."
)

# The Artel team's coach. The baseline brain is good but STATIC — it plays the same way at 0-0 and
# 1-0 with five minutes left, and it never targets a specific opponent's weakness. The coach reads
# the live game every window and re-throws the plan: attack the opponent's weakest channel, and
# commit or hold numbers by the scoreline and clock. That adaptation is the edge a fixed team can't
# answer. The plan is authored by an LLM coach when one is configured (PITCH_LLM_KEY), else by the
# proven heuristic; either way the plan + line pieces are published to and read back from Artel.


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


def combined_brain(coords: dict):
    # route each player to its side's coordinator (Artel sides) or the baseline. With half the field
    # Artel, a match can have two coordinators — both teams coached, each by its own line agents.
    def brain(pitch: Pitch, p: Player) -> dict:
        co = coords.get(p.team)
        return coordinated_decide(pitch, p, co.plan) if co else bot.decide(pitch, p)

    return brain


def _state_prompt(pitch: Pitch, team: str) -> str:
    c = pitch.cfg
    other = "away" if team == "home" else "home"
    base = plan_for(pitch, team)  # reuse the heuristic read of the weak flank
    secs = int(pitch.tick / c.match_ticks * 5400)
    weak = "left" if base.overload_y < c.width / 2 else "right"
    ours = "-".join(map(str, pitch.shapes.get(team, ())))
    theirs = "-".join(map(str, pitch.shapes.get(other, ())))
    return (
        f"Score: us {pitch.score[team]}, them {pitch.score[other]}. "
        f"Clock {secs // 60}:{secs % 60:02d} of 90 (half {pitch.half}). "
        f"Our shape {ours}, their shape {theirs}. "
        f"Their weaker defensive flank looks {weak}. Set the plan."
    )


async def author_plan_llm(pitch: Pitch, team: str) -> Plan:
    fallback = plan_for(pitch, team)
    out = llm.parse_json(await llm.complete(COACH_SYSTEM, _state_prompt(pitch, team)))
    if not out:
        return fallback
    c = pitch.cfg
    lanes = {"left": c.width * 0.27, "right": c.width * 0.73, "center": c.width * 0.5}
    overload_y = lanes.get(str(out.get("overload", "")).lower(), fallback.overload_y)
    try:
        commit = max(0, min(3, int(out.get("commit", fallback.commit))))
    except (TypeError, ValueError):
        commit = fallback.commit
    return Plan(
        overload_y=overload_y,
        commit=commit,
        low_block=bool(out.get("low_block", fallback.low_block)),
    )


def _parse_plan(mid_rows: list, fwd_rows: list, fallback: Plan) -> Plan:
    # reassemble the Plan from what the line agents posted to Artel memory
    p = Plan(fallback.overload_y, fallback.commit, fallback.low_block)
    for row in mid_rows:
        for tok in str(row.get("content", "")).split(";"):
            if tok.startswith("commit="):
                p.commit = int(tok.split("=")[1])
            elif tok.startswith("low_block="):
                p.low_block = tok.split("=")[1] == "1"
    for row in fwd_rows:
        for tok in str(row.get("content", "")).split(";"):
            if tok.startswith("overload_y="):
                p.overload_y = float(tok.split("=")[1])
    return p


class Coordinator:
    """The Artel team's coaching staff: a captain plus a defence / midfield / attack agent, each a
    distinct Artel identity. The captain sets the lineup once; each window the line agents post
    their piece of the plan to Artel and the team then executes the plan read BACK from Artel — so
    the coordination genuinely flows through the server. Falls back to a local plan if Artel is
    unconfigured or unreachable, so a match never depends on the network."""

    def __init__(self, team: str) -> None:
        self.team = team
        self.plan = Plan(overload_y=40.0, commit=0, low_block=False)
        self._last = Plan(overload_y=-1, commit=-1, low_block=False)
        self._artel = artel_client.Artel() if artel_client.configured() else None
        self.live = self._artel is not None  # genuinely talking to Artel this match
        self.llm = llm.enabled()  # the coach is an LLM (else the deterministic heuristic)
        self._llm_at = 0.0
        self._llm_plan: Plan | None = None
        self._llm_busy = False

    async def _author(self, pitch: Pitch) -> Plan:
        # the coach's plan: an LLM authors it on a slow cadence (one call in flight, deterministic
        # fallback); without an LLM it's the proven heuristic, recomputed every window.
        if not self.llm:
            return plan_for(pitch, self.team)
        now = time.monotonic()
        if not self._llm_busy and now - self._llm_at >= LLM_EVERY:
            self._llm_busy = True
            self._llm_at = now
            try:
                self._llm_plan = await author_plan_llm(pitch, self.team)
            finally:
                self._llm_busy = False
        return self._llm_plan or plan_for(pitch, self.team)

    def optimize(self, pitch: Pitch) -> None:
        # captain's team-sheet — must run before the first tick, so it's synchronous
        optimize_lineup([p for p in pitch.players if p.team == self.team])

    async def announce(self) -> None:
        # captain posts the match plan to Artel shared memory (best-effort)
        if self._artel:
            await self._artel.write_memory(
                "captain",
                "match plan: attack the opponent's weak channel; commit when behind, sit on a lead",
                ["pitch", "plan", "match"],
            )

    async def refresh(self, pitch: Pitch) -> None:
        base = await self._author(pitch)  # LLM coach if configured, else the proven heuristic
        if self._artel is None:
            self.plan = base
            return
        # line agents publish their piece to Artel, tagged by this team so two coordinators in an
        # Artel-vs-Artel tie don't read each other's plan (only on change, to keep traffic light).
        tag = f"team:{self.team}"
        if base.commit != self._last.commit or base.low_block != self._last.low_block:
            await self._artel.write_memory(
                "mid",
                f"commit={base.commit};low_block={int(base.low_block)}",
                ["pitch", "line", tag],
            )
        if round(base.overload_y) != round(self._last.overload_y):
            await self._artel.write_memory(
                "fwd", f"overload_y={base.overload_y}", ["pitch", "line", tag]
            )
            await self._artel.emit_event(
                "fwd", "overload", {"team": self.team, "y": round(base.overload_y)}
            )
        self._last = base
        # the team executes the plan as read BACK from Artel — the line agents read each other's
        # posts (filtered to our team) and reassemble the plan: genuine coordination through Artel
        rows = await self._artel.search_memory("def", "commit overload", tag=tag, limit=4)
        self.plan = _parse_plan(rows, rows, base)

    async def aclose(self) -> None:
        if self._artel:
            await self._artel.aclose()

    def brain(self):
        def b(pitch: Pitch, p: Player) -> dict:
            if p.team == self.team:
                return coordinated_decide(pitch, p, self.plan)
            return bot.decide(pitch, p)

        return b
