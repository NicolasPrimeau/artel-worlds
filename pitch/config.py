from __future__ import annotations

from dataclasses import dataclass


# Small-sided soccer on a continuous 2D pitch — 5-a-side (a keeper + four outfield), because
# fewer players means more goals, faster play, and coordination that READS at a glance (the
# whole point: a team that moves as a unit vs. one that doesn't should be obvious without an
# overlay). Home attacks toward +x (the right goal); away toward -x (the left goal). Units are
# abstract "metres"; a tick is one simulation step.
@dataclass(frozen=True)
class Config:
    length: float = 120.0  # goal-to-goal
    width: float = 80.0  # touchline-to-touchline
    goal_width: float = 24.0  # span of the goal mouth, centred on the width

    team_size: int = 9  # includes the keeper — a 3-3-2; bump for an even fuller pitch

    player_speed: float = 0.82  # max outfield move per tick — calmer, more deliberate pace
    keeper_speed: float = 1.25  # keepers are quick across the mouth (they make the saves)
    accel: float = 0.15  # gentler ramp to top speed — players glide and arc, not snap
    arrive_radius: float = (
        4.0  # ease to a stop within this of a POSITIONING target (not when chasing)
    )

    ball_friction: float = 0.965  # ball velocity retained per tick when loose
    shot_speed: float = 3.3
    pass_speed: float = 2.7  # crisp — reaches the receiver before a defender can step in
    dribble_speed: float = 1.9  # a carried ball is nudged forward at ~player pace
    control_radius: float = 2.4  # within this of the ball, a player is "on" it
    tackle_radius: float = 2.2  # an opponent this close can contest possession
    gk_reach: float = 3.9  # a keeper gathers (saves) the ball within this — bigger than control

    shoot_range: float = 31.0  # shoot only when genuinely close — forces build-up, not blasts
    pass_lead: float = 2.5  # lead a teammate by this much — small, so the ball arrives to feet
    carry_ahead: float = 1.6  # a dribbled ball rides this far ahead of the carrier's feet
    carry_ease: float = (
        0.5  # how snappily the carried ball tracks that spot (0..1) — smooth, not glued
    )

    celebrate_ticks: int = 32  # freeze on a goal so the score is readable before kickoff
    restart_ticks: int = 6  # brief dead-ball pause on a throw-in / corner / goal kick
    halftime_ticks: int = 50  # the break between the two halves

    match_ticks: int = 2400  # length of one match (two halves of match_ticks // 2)


DEFAULT = Config()
