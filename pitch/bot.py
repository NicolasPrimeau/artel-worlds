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


def _offside_adv(pitch: Pitch, p: Player) -> float:
    # the offside line in attacking-direction terms: the second-to-last opponent's advance. A forward
    # should hold on this line (the last defender's shoulder), not loiter beyond it in the keeper's zone.
    advs = sorted((_fwd_x(p.team, o.x, pitch.cfg.length) for o in pitch.opponents(p)), reverse=True)
    return advs[1] if len(advs) >= 2 else (advs[0] if advs else pitch.cfg.length)


def _pursuit_cost(pitch: Pitch, q: Player, b) -> float:
    # who presses the ball: nearest, but a player pays a penalty for leaving their zone — so a
    # defender doesn't charge the attacking third and a striker doesn't track into his own box.
    L = pitch.cfg.length
    fx = _fwd_x(q.team, b.x, L)
    ball_zone = 0 if fx < L / 3 else (1 if fx < 2 * L / 3 else 2)
    return _len(q.x - b.x, q.y - b.y) + abs(ball_zone - ROLE_ZONE.get(q.role, 1)) * 20.0


def _mark_target(pitch: Pitch, p: Player) -> tuple[float, float]:
    # when the opponent has the ball, don't just hold a zone — pick up the nearest attacker in our
    # area and sit just goal-side of them, denying the easy pass. This is what forces the ball into
    # tight spaces where passes get read and intercepted. Blended with shape so we don't over-commit.
    c = pitch.cfg
    base = _formation_target(pitch, p)
    foes = [o for o in pitch.opponents(p) if o.role != "GK"]
    if not foes:
        return base
    foe = min(foes, key=lambda o: _len(o.x - p.x, o.y - p.y))
    if _len(foe.x - p.x, foe.y - p.y) > 16.0:  # nobody near enough to mark — just hold shape
        return base
    own_goal_x = 0.0 if p.team == "home" else c.length
    ux, uy = _unit(own_goal_x - foe.x, c.width / 2 - foe.y)
    # sit goal-side but a few steps OFF — close enough to contest a pass, loose enough that the man
    # is still a viable receiver, so the ball moves instead of every pass dying in traffic
    mx, my = foe.x + ux * 4.0, foe.y + uy * 4.0
    return (
        _clamp(mx * 0.6 + base[0] * 0.4, 6.0, c.length - 6.0),
        _clamp(my * 0.6 + base[1] * 0.4, 6.0, c.width - 6.0),
    )


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
        # forwards hold a high line, but stay ONSIDE — on the last defender's shoulder, not loitering
        # beyond it in the keeper's zone (which is offside anyway and makes a useless pass target)
        fadv = max(_fwd_x(p.team, fx, L), L * 0.42)
        fadv = min(fadv, _offside_adv(pitch, p) - 1.0)
        fx = fadv if p.team == "home" else L - fadv
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

    # SET PIECE: on a corner, attackers crash the box and defenders pack it (keeper holds its line)
    if pitch.restart_kind == "corner" and pitch.possessor != p.id and p.role != "GK":
        badv = _fwd_x(p.team, b.x, c.length)  # how advanced the ball is for p's team
        spread = _clamp(c.width / 2 + ((p.id % 5) - 2) * 7.0, 8.0, c.width - 8.0)
        if badv > c.length * 0.70 and p.role in ("FWD", "MID"):  # attacking — crash the box
            return {"move": (c.length - 12 if p.team == "home" else 12, spread)}
        if badv < c.length * 0.30 and p.role in ("DEF", "MID"):  # defending — pack the box
            return {"move": (12 if p.team == "home" else c.length - 12, spread)}

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
        # CROSS from wide in the final third (or taking a corner) — whip it into the box for a
        # team-mate to attack. A big share of real goals come from balls into the box.
        adv_me = _fwd_x(p.team, p.x, c.length)
        wide = abs(p.y - c.width / 2) > c.width * 0.30
        target_in_box = [
            q
            for q in outfield
            if q.id != p.id
            and _fwd_x(q.team, q.x, c.length) > c.length * 0.78
            and abs(q.y - c.width / 2) < 22
        ]
        if target_in_box and (
            pitch.restart_kind == "corner" or (wide and adv_me > c.length * 0.74)
        ):
            box_x = c.length - 11 if p.team == "home" else 11
            box_y = _clamp(c.width / 2 + rng.choice((-1, 1)) * 6, 8, c.width - 8)
            return _kick(p, box_x, box_y, c.pass_speed * 1.15, 0.14, rng)
        # Keep the ball moving — but FORWARD. Pass often (retain possession), strongly favouring the
        # team-mate who advances the play, so the ball climbs into the final third instead of going
        # square forever. Carry at goal when no pass is on. This is what turns possession into shots.
        mates = [q for q in outfield if q.id != p.id]
        opts = [q for q in mates if _open(pitch, q) > 3.5 and (q.x - p.x) * fwd > -2]
        pressured = mine_open < 6.0
        if opts and (pressured or len(opts) >= 2 or rng.random() < 0.6):
            tgt = max(opts, key=lambda q: (q.x - p.x) * fwd * 0.6 + _open(pitch, q) * 0.4)
            noise = 0.08 + max(0.0, 8.0 - _open(pitch, tgt)) * 0.012
            return _kick(p, tgt.x + c.pass_lead * fwd, tgt.y, c.pass_speed, noise, rng)
        # carry: drive at goal down the open flank, sprinting into clear space to reach shooting range
        return {
            "move": _attack_target(pitch, p),
            "sprint": mine_open > 8.0 and dist_goal > c.shoot_range,
        }

    if pitch.possessor is not None and pitch.possessor in teammate_ids:
        # a teammate has the ball — don't chase our own player; hold shape and offer support
        return {"move": _formation_target(pitch, p), "kick": None}

    # loose ball or the opponent has it — the role-appropriate nearest presses, and a SECOND player
    # closes down if they're near (a real press unit that actually wins the ball / cuts out passes),
    # while the rest hold their line. Two pressers, not a swarm.
    ranked = sorted(outfield, key=lambda q: _pursuit_cost(pitch, q, b))
    pressers = {ranked[0].id}
    if len(ranked) > 1 and _len(ranked[1].x - b.x, ranked[1].y - b.y) < 20.0:
        pressers.add(ranked[1].id)
    if p.id in pressers:
        return {"move": (b.x, b.y), "sprint": True}
    # not pressing: mark a man if the opponent has the ball, else just hold shape (loose ball)
    opponent_has_it = pitch.possessor is not None and pitch.possessor not in teammate_ids
    target = _mark_target(pitch, p) if opponent_has_it else _formation_target(pitch, p)
    return {"move": target, "kick": None}
