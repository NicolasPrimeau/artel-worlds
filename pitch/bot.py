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


def _attack_target(pitch: Pitch, p: Player) -> tuple[float, float]:
    # drive at goal from the MORE OPEN flank, not always straight up the middle — attacks come in
    # at varied angles depending on where the defence is overloaded.
    c = pitch.cfg
    gx, _gy = pitch.attack_goal(p.team)
    foes = pitch.opponents(p)
    top = sum(1 for o in foes if o.y < c.width / 2)
    bot = sum(1 for o in foes if o.y >= c.width / 2)
    lane = c.width * 0.30 if top <= bot else c.width * 0.70
    ty = _clamp(p.y * 0.45 + lane * 0.55, 8.0, c.width - 8.0)
    return gx, ty


def _kick(
    p: Player, tx: float, ty: float, speed: float, noise: float, rng, skill: float = 0.0
) -> dict:
    ux, uy = _unit(tx - p.x, ty - p.y)
    # accuracy noise: perturb the heading (more for harder kicks); a sharper player scatters less.
    # `skill` is the relevant rating (passing for passes, finishing for shots). deterministic.
    sk = skill or p.acc
    a = (rng.random() - 0.5) * noise * (2.0 - sk)
    cs = 1 - a * a / 2
    sn = a
    rx, ry = ux * cs - uy * sn, ux * sn + uy * cs
    return {"move": (tx, ty), "kick": (rx * speed, ry * speed)}


ROLE_ZONE = {"DEF": 0, "MID": 1, "FWD": 2}  # which third of the pitch a role belongs in


def _fwd_x(team: str, x: float, length: float) -> float:
    # x measured up the attacking direction: 0 = own goal line, length = opponent's goal line
    return x if team == "home" else length - x


def _pursuit_cost(pitch: Pitch, q: Player, b) -> float:
    # who presses the ball: nearest, but a player pays a penalty for leaving their zone — so a
    # defender doesn't charge the attacking third and a striker doesn't track into his own box.
    L = pitch.cfg.length
    fx = _fwd_x(q.team, b.x, L)
    ball_zone = 0 if fx < L / 3 else (1 if fx < 2 * L / 3 else 2)
    return _len(q.x - b.x, q.y - b.y) + abs(ball_zone - ROLE_ZONE.get(q.role, 1)) * 20.0


def _formation_target(pitch: Pitch, p: Player) -> tuple[float, float]:
    c = pitch.cfg
    b = pitch.ball
    L = c.length
    # team shape slides toward the ball's x; forwards push further than defenders.
    role_push = {"DEF": 0.35, "MID": 0.7, "FWD": 1.0}.get(p.role, 0.5)
    shift = (b.x - L / 2) * role_push
    fx = p.home_x + shift
    fwd_ball = _fwd_x(p.team, b.x, L)  # how deep into our half the ball is (0 = our goal line)
    fy = p.home_y * 0.55 + b.y * 0.45  # hold width but lean to the ball's side
    if p.role == "DEF":
        # defenders stay goal-side of the ball and never drift into the attacking half — a back line
        fx = min(fx, b.x - 4) if p.team == "home" else max(fx, b.x + 4)
        fx = min(fx, L * 0.52) if p.team == "home" else max(fx, L * 0.48)
        if fwd_ball < L * 0.4:
            # ball threatening our third — drop deep and slide to the ball's side to cover the
            # flank the attack is coming down, rather than sitting centrally and leaving it open
            goal_x = 0.0 if p.team == "home" else L
            fx = fx * 0.4 + (goal_x + (15 if p.team == "home" else -15)) * 0.6
            fy = fy * 0.35 + b.y * 0.4 + (c.width / 2) * 0.25
    elif p.role == "FWD":
        # forwards hold a high line — they stay an outlet up top even when the ball is deep
        fx = max(fx, L * 0.42) if p.team == "home" else min(fx, L * 0.58)
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
    teammate_ids = {q.id for q in pitch.teammates(p)}

    if pitch.possessor == p.id:
        # ON THE BALL: shoot (only when genuinely close), else PASS by default, else carry.
        fwd = 1.0 if p.team == "home" else -1.0
        dist_goal = _len(gx - p.x, gy - p.y)
        mine_open = _open(pitch, p)
        # shoot when in range and with a sight of goal — and ALWAYS when point-blank, since you
        # can't just carry it over the line: a goal has to be struck.
        if dist_goal < c.shoot_range and (mine_open > 3.0 or dist_goal < 13.0):
            ax, ay = _shoot_aim(pitch, p)
            # scatter grows with range — long shots fly wide, so goals come from working it close
            return _kick(p, ax, ay, c.shot_speed, 0.22 + dist_goal / 130.0, rng, skill=p.finishing)
        # PASS is the default — keep it moving. Favour a teammate who is OPEN and REACHABLE (a
        # short, completable ball beats a hopeful long one); a little forward progress is a bonus.
        # This is what makes the build-up actually connect instead of breaking down on attack.
        mates = [q for q in outfield if q.id != p.id]
        opts = [q for q in mates if _open(pitch, q) > 5.0 and (q.x - p.x) * fwd > -8]
        pressured = mine_open < 6.5
        if opts and (pressured or len(opts) >= 2 or rng.random() < 0.6):
            tgt = max(
                opts,
                key=lambda q: _open(pitch, q) * 0.8
                + (q.x - p.x) * fwd * 0.35
                - _len(q.x - p.x, q.y - p.y) * 0.3,
            )
            return _kick(p, tgt.x + c.pass_lead * fwd, tgt.y, c.pass_speed, 0.05, rng)
        # carry: drive at goal via the more open flank — NO kick, so the engine eases the ball ahead
        # of the carrier (a smooth dribble) and they jink around defenders on the way.
        return {"move": _attack_target(pitch, p)}

    if pitch.possessor is not None and pitch.possessor in teammate_ids:
        # a teammate has the ball — don't chase our own player; hold shape and offer support
        return {"move": _formation_target(pitch, p), "kick": None}

    # loose ball or the opponent has it — only the role-appropriate nearest player presses; the
    # rest hold their line. A defender wins it in our third, a forward leads the press up high.
    pursuer = min(outfield, key=lambda q: _pursuit_cost(pitch, q, b))
    if pursuer.id == p.id:
        return {"move": (b.x, b.y), "sprint": True}
    return {"move": _formation_target(pitch, p), "kick": None}
