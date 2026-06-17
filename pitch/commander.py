from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass

from . import artel_client, bot, llm, plays
from .engine import Pitch, Player, _clamp

LLM_EVERY = float(os.environ.get("PITCH_LLM_EVERY", "6"))  # seconds between coach LLM calls
PLAN_TTL = float(os.environ.get("PITCH_PLAN_TTL", "12"))  # a directive expires this long after the
# last successful read — so the coordination edge can't outlive the Artel bus that carries it
DIRECTIVE = "pitch.directive"

COACH_SYSTEM = (
    "You are the head coach of an AI soccer team in a live 2D match. Read the situation and set the "
    "game plan. Reply with ONLY a JSON object, no prose: "
    '{"overload":"left|right|center","commit":0-3,"low_block":true|false,"combos":true|false}. '
    "overload = the flank to attack (pick the opponent's WEAKER side). "
    "Game management is what wins — DO NOT over-commit and get countered. Follow these rules: "
    "If AHEAD in the last third of the match: low_block=true, commit=0 (protect the lead). "
    "If LOSING: commit 2-3 and push (more the later it is); low_block=false. "
    "If LEVEL: commit 0-1 and stay balanced; low_block=false. "
    "combos: true only when their defence is packed/deep, else false."
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
    combos: bool = False  # call quick give-and-go combinations (the play actuator acts on this)


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
                base = bot._formation_target(pitch, p)
                py = _clamp(base[1] * 0.45 + plan.overload_y * 0.55, 8, c.width - 8)
                return {"move": (push, py), "kick": None}
        if p.role == "FWD":
            # forwards lead the line, LEANING into the weak channel — but keep their spread (blend
            # with the formation slot) so they don't all stack on one marked spot
            base = bot._formation_target(pitch, p)
            fy = _clamp(base[1] * 0.5 + plan.overload_y * 0.5, 8, c.width - 8)
            return {"move": (base[0], fy), "kick": None}

    return bot.decide(pitch, p)  # defend / press / hold — all baseline


def combined_brain(coords: dict):
    # route each player to its side's coordinator or the baseline. The coordination EDGE only applies
    # when a live Artel directive is in hand (co.plan set); with no directive — Artel down, LLM not
    # configured, or none authored yet — the side plays pure baseline. So the edge is genuinely
    # Artel-borne: cut the bus and the coached team is indistinguishable from the rest.
    st = {"tick": -1}

    def brain(pitch: Pitch, p: Player) -> dict:
        if pitch.tick != st["tick"]:  # advance/trigger each side's plays once per tick
            st["tick"] = pitch.tick
            for co in coords.values():
                co.plays.update(pitch)
        co = coords.get(p.team)
        if co is None:
            return bot.decide(pitch, p)
        it = co.plays.intent(pitch, p)  # the play actuator overrides for the players in a play
        if it is not None:
            return it
        if co.plan is None:
            return bot.decide(pitch, p)
        return coordinated_decide(pitch, p, co.plan)

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
        combos=bool(out.get("combos", False)),
    )


def _now_cursor() -> str:
    # an Artel-event-format timestamp for "now", so a directive read only sees THIS match's calls
    n = datetime.datetime.now(datetime.UTC)
    return n.strftime("%Y-%m-%dT%H:%M:%S.") + f"{n.microsecond // 1000:03d}Z"


def _say_directive(club: str, plan: Plan, side: str) -> str:
    # a human-readable line of the coach's call, so the Artel project reads as real coaching
    if plan.low_block:
        return f"{club}: low block — protect the lead, stay compact and deny the space."
    if plan.commit >= 2:
        return f"{club}: overload the {side}, commit {plan.commit} runners — chase the game."
    if plan.commit == 1:
        return f"{club}: work the {side} flank and push a runner up in support."
    return f"{club}: attack down the {side}, hold our shape."


class Coordinator:
    """The Artel team's coach — an LLM tactical agent. Each window it reads the live match and
    authors a coordinated directive (overload a flank, commit runners, drop into a low block), then
    PUBLISHES it to Artel; the team plays whatever directive Artel currently holds, read straight
    back out. There is no local fallback for the directive: no Artel, no configured LLM, or no call
    yet => plan is None and the team plays pure baseline. So the coordination EDGE genuinely rides
    on Artel — cut the bus (or the model) and the directive expires and the side is just a baseline
    team again. Per-player execution stays deterministic; the LLM only sets the tactical policy."""

    def __init__(self, team: str, club: str = "") -> None:
        self.team = team
        self.club = club or team
        self.plan: Plan | None = None  # active directive; None = no edge -> baseline play
        self._artel = artel_client.Artel() if artel_client.configured() else None
        self.live = self._artel is not None
        self.llm = llm.enabled()  # is an LLM coach configured to author directives?
        self._llm_at = 0.0
        self._busy = False
        self._since = _now_cursor()
        self._plan_at = 0.0  # monotonic time the current directive was last read (for TTL expiry)
        self.plays = plays.PlayManager(
            team
        )  # the actuator: runs plays when the directive calls them

    def optimize(self, pitch: Pitch) -> None:
        # captain's team-sheet — fit the rolled attributes to roles before kickoff (synchronous)
        optimize_lineup([p for p in pitch.players if p.team == self.team])

    async def announce(self) -> None:
        # nothing to pre-stage: live coordination flows as directive EVENTS, not memories
        return

    async def refresh(self, pitch: Pitch) -> None:
        if self._artel is None:
            self.plan = None  # no bus -> no edge
            return
        now = time.monotonic()
        # the LLM coach authors a fresh directive on a slow cadence and PUBLISHES it to Artel
        if self.llm and not self._busy and now - self._llm_at >= LLM_EVERY:
            self._busy = True
            self._llm_at = now
            try:
                plan = await author_plan_llm(pitch, self.team)
                await self._publish(pitch, plan)
            finally:
                self._busy = False
        # the team's plan IS whatever directive Artel hands back; expire it if none arrives in TTL so
        # the edge can't outlive the bus (Artel/LLM down => no fresh directive => baseline)
        got = await self._read()
        if got is not None:
            self.plan, self._plan_at = got, now
        elif now - self._plan_at > PLAN_TTL:
            self.plan = None
        # hand the actuator its standing play-call from the (Artel-borne) directive — no plan, no call
        if self.plan is not None:
            side = "left" if self.plan.overload_y < pitch.cfg.width / 2 else "right"
            self.plays.call = {"combos": self.plan.combos, "channel": side}
        else:
            self.plays.call = None

    async def _publish(self, pitch: Pitch, plan: Plan) -> None:
        # the directive is an EVENT (the right primitive for live coordination — ephemeral, pub/sub,
        # not a memory). It carries the structured plan the team reads back AND a readable line.
        side = "left" if plan.overload_y < pitch.cfg.width / 2 else "right"
        await self._artel.emit_event(
            "captain",
            DIRECTIVE,
            {
                "team": self.team,
                "overload_y": round(plan.overload_y, 1),
                "commit": plan.commit,
                "low_block": plan.low_block,
                "combos": plan.combos,
                "say": _say_directive(self.club, plan, side),
            },
        )

    async def _read(self) -> Plan | None:
        rows = await self._artel.poll_events("captain", DIRECTIVE, self._since)
        latest = None
        for r in rows:
            self._since = max(self._since, r.get("created_at", self._since))
            p = r.get("payload", {})
            if p.get("team") == self.team:
                latest = p
        if latest is None:
            return None
        return Plan(
            float(latest["overload_y"]),
            int(latest["commit"]),
            bool(latest["low_block"]),
            bool(latest.get("combos", False)),
        )

    async def finish(self, summary: str) -> None:
        # record the result; KEEP the readable directive log (it's sparse and is the visible proof
        # that coaching flowed through Artel), only sweep the ephemeral tasks/messages
        if self._artel:
            await self._artel.emit_event("captain", "pitch.result", {"result": summary})

    async def aclose(self) -> None:
        if self._artel:
            await self._artel.aclose()

    def brain(self):
        def b(pitch: Pitch, p: Player) -> dict:
            if p.team == self.team and self.plan is not None:
                return coordinated_decide(pitch, p, self.plan)
            return bot.decide(pitch, p)

        return b
