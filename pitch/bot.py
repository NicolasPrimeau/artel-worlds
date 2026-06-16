from __future__ import annotations

from .engine import Pitch, Player, _clamp, _len, _unit


# The baseline soccer brain — the SAME deterministic motor both teams run, so any divergence the
# arena shows is coordination, not a head start (the phalanx invariant). The one rule that turns
# 22 toddlers chasing a ball into something that reads like soccer: only the nearest teammate
# pursues; everyone else holds team shape. Later, an Artel commander overrides shape/role/press
# on ONE team — that's the whole experiment.


def _shoot_aim(pitch: Pitch, p: Player) -> tuple[float, float]:
    gx, gy = pitch.attack_goal(p.team)
    side = pitch._rng.choice((-1, 1))  # pick a corner of the mouth
    aim_y = gy + side * pitch.cfg.goal_width * 0.34
    return gx, _clamp(aim_y, gy - pitch.cfg.goal_width / 2, gy + pitch.cfg.goal_width / 2)


def _ahead(team: str, q: Player, p: Player) -> bool:
    # is q closer to the attacking goal than p (a forward option)?
    return q.x > p.x + 1.0 if team == "home" else q.x < p.x - 1.0


def _open(pitch: Pitch, q: Player) -> float:
    # distance to the nearest opponent — bigger is more open
    return min((_len(o.x - q.x, o.y - q.y) for o in pitch.opponents(q)), default=99.0)


def _kick(p: Player, tx: float, ty: float, speed: float, noise: float, rng) -> dict:
    ux, uy = _unit(tx - p.x, ty - p.y)
    # accuracy noise: perturb the heading (more for harder kicks); deterministic via the rng
    a = (rng.random() - 0.5) * noise
    cs = 1 - a * a / 2
    sn = a
    rx, ry = ux * cs - uy * sn, ux * sn + uy * cs
    return {"move": (tx, ty), "kick": (rx * speed, ry * speed)}


def _formation_target(pitch: Pitch, p: Player) -> tuple[float, float]:
    c = pitch.cfg
    b = pitch.ball
    # team shape slides toward the ball's x; forwards push further than defenders.
    role_push = {"DEF": 0.35, "MID": 0.7, "FWD": 1.0}.get(p.role, 0.5)
    shift = (b.x - c.length / 2) * role_push
    fx = p.home_x + shift
    # defenders never get caught upfield of the ball — stay goal-side
    if p.role == "DEF":
        fx = min(fx, b.x - 4) if p.team == "home" else max(fx, b.x + 4)
    fy = p.home_y * 0.55 + b.y * 0.45  # hold width but lean to the ball's side
    fx = _clamp(fx, 8.0, c.length - 8.0)
    fy = _clamp(fy, 6.0, c.width - 6.0)
    # de-stack: ease away from the nearest teammate so two players never occupy a spot
    near = min(
        (q for q in pitch.teammates(p) if q.id != p.id),
        key=lambda q: _len(q.x - fx, q.y - fy),
        default=None,
    )
    if near is not None and _len(near.x - fx, near.y - fy) < 7.0:
        ax, ay = _unit(fx - near.x, fy - near.y)
        fx, fy = fx + ax * 4, fy + ay * 4
    return _clamp(fx, 4.0, c.length - 4.0), _clamp(fy, 4.0, c.width - 4.0)


def decide(pitch: Pitch, p: Player) -> dict:
    c = pitch.cfg
    b = pitch.ball
    rng = pitch._rng
    gx, gy = pitch.attack_goal(p.team)
    own_x = 0.0 if p.team == "home" else c.length

    if p.role == "GK":
        if pitch.possessor == p.id:  # gathered it — clear upfield to a wide outlet
            tx = c.length * 0.62 if p.team == "home" else c.length * 0.38
            ty = c.width * (0.28 if rng.random() < 0.5 else 0.72)
            return _kick(p, tx, ty, c.shot_speed * 0.92, 0.18, rng)
        keep_x = own_x + (7.0 if p.team == "home" else -7.0)
        # ANTICIPATE the shot: slide to where the ball will cross the keeper's line, not just to
        # the ball's current y — a keeper that's actually in the way of the shot. Sprint when the
        # ball is in our third so we get there in time and gather it (a save).
        ty = b.y
        toward = (b.vx < -0.05) if p.team == "home" else (b.vx > 0.05)
        if toward:
            t = (keep_x - b.x) / b.vx
            if 0 < t < 60:
                ty = b.y + b.vy * t
        ty = _clamp(ty, c.width / 2 - c.goal_width / 2, c.width / 2 + c.goal_width / 2)
        threat = (b.x < c.length * 0.34) if p.team == "home" else (b.x > c.length * 0.66)
        return {"move": (keep_x, ty), "sprint": threat}

    outfield = [q for q in pitch.teammates(p) if q.role != "GK"]
    pursuer = min(outfield, key=lambda q: _len(q.x - b.x, q.y - b.y))

    if pursuer.id != p.id:  # not my ball — hold shape
        return {"move": _formation_target(pitch, p), "kick": None}

    if pitch.possessor != p.id:  # chase / intercept — sprint to win the ball
        return {"move": (b.x, b.y), "sprint": True}

    # on the ball: shoot, pass, or carry
    dist_goal = _len(gx - p.x, gy - p.y)
    if dist_goal < c.shoot_range:
        ax, ay = _shoot_aim(pitch, p)
        return _kick(p, ax, ay, c.shot_speed, 0.12 + dist_goal / 400.0, rng)

    # best forward, open pass
    options = [q for q in outfield if q.id != p.id and _ahead(p.team, q, p)]
    options = [q for q in options if _open(pitch, q) > 5.0]
    if options:
        tgt = max(options, key=lambda q: (q.x if p.team == "home" else -q.x) + _open(pitch, q))
        lead_x = tgt.x + (c.pass_lead if p.team == "home" else -c.pass_lead)
        return _kick(p, lead_x, tgt.y, c.pass_speed, 0.10, rng)

    # carry: drive at goal down the more open lane — NO kick, so the engine eases the ball
    # ahead of us (a smooth dribble that the client can interpolate), instead of re-striking it.
    push_y = _clamp(p.y + rng.choice((-1, 1)) * 5.0, 6, c.width - 6)
    return {"move": (gx, push_y)}
