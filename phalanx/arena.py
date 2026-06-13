from __future__ import annotations

import random

from .config import Config
from .tank import AXIAL_DIRS, Tank, dir_toward, hex_distance, hex_line

_TEAM_NAMES = ("artel", "red", "green", "gold", "violet", "cyan")
_VALID_MOVE = ("fwd", "back", "hold")


class Arena:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tick_count = 0
        self._next_id = 0
        self.tanks: dict[int, Tank] = {}
        self.tracers: list[dict] = []  # transient shots fired this tick, for the viz
        self.events: list[str] = []  # kill log — real match facts for after-action lessons
        self.team_kind: dict[str, str] = {}  # team -> "house:<name>" | "player:<agent>"
        self.pending: dict[int, dict] = {}
        self.walls: set[tuple[int, int]] = set()
        self.winner: str | None = None
        self.draw = False
        # per-team match counters — every measured fact here is a family of after-action
        # rules the reflector can legitimately derive; an unmeasured one cannot exist
        self.match_stats: dict[str, dict[str, float]] = {}
        self._scatter_walls()

    # --- setup ---
    def cells(self) -> list[tuple[int, int]]:
        R = self.cfg.map_radius
        return [
            (q, r)
            for q in range(self.cfg.width)
            for r in range(self.cfg.height)
            if hex_distance(q, r, R, R) <= R
        ]

    def _free_connected(self) -> bool:
        free = {c for c in self.cells() if c not in self.walls}
        if not free:
            return False
        start = next(iter(free))
        seen = {start}
        stack = [start]
        while stack:
            q, r = stack.pop()
            for dq, dr in AXIAL_DIRS:
                n = (q + dq, r + dr)
                if n in free and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == len(free)

    def _scatter_walls(self) -> None:
        # Place cover one block at a time, keeping only blocks that leave the open space
        # fully connected — so a wall line can never split the arena into sealed halves.
        R = self.cfg.map_radius
        interior = [c for c in self.cells() if hex_distance(c[0], c[1], R, R) <= R - 2]
        target = int(len(self.cells()) * self.cfg.obstacle_density)
        attempts = 0
        while len(self.walls) < target and attempts < target * 25:
            attempts += 1
            q, r = self.rng.choice(interior)
            if (q, r) in self.walls:
                continue
            self.walls.add((q, r))
            if not self._free_connected():
                self.walls.discard((q, r))  # would wall off part of the map — reject

    def _in_bounds(self, q: int, r: int) -> bool:
        R = self.cfg.map_radius
        return hex_distance(q, r, R, R) <= R  # the arena is a hexagon, not a box

    def _free_cell(self, near: tuple[int, int] | None = None) -> tuple[int, int]:
        # spawns CLUSTER tight around the anchor (radius 1) — a team starts as a formed
        # unit, not a scattered one; the anchors themselves sit far apart
        for _ in range(200):
            if near is not None:
                q = near[0] + self.rng.randint(-1, 1)
                r = near[1] + self.rng.randint(-1, 1)
            else:
                q, r = self.rng.choice(self.cells())
            if self._in_bounds(q, r) and (q, r) not in self.walls and not self._tank_at(q, r):
                return (q, r)
        return self.rng.choice([c for c in self.cells() if c not in self.walls])

    def add_team(self, team: str, controller: str, anchor: tuple[int, int]) -> list[int]:
        self.team_kind[team] = controller
        ids = []
        for _ in range(self.cfg.team_size):
            q, r = self._free_cell(near=anchor)
            ids.append(self._spawn(team, controller, q, r))
        return ids

    def _spawn(self, team: str, controller: str, q: int, r: int) -> int:
        self._next_id += 1
        self.tanks[self._next_id] = Tank(
            id=self._next_id,
            team=team,
            q=q,
            r=r,
            heading=self.rng.randint(0, 5),
            energy=float(self.cfg.start_energy),
            controller=controller,
        )
        return self._next_id

    def seed_house(self, flip: bool = False) -> None:
        # Spawns point-symmetric about the control center. A residual corner edge still
        # exists (fixed turn chirality), so the caller flips corners every match — each
        # team spends half its matches in each corner and the edge cancels, leaving
        # Artel as the only systematic variable across the series.
        cq, cr = self.cfg.width // 2, self.cfg.height // 2
        corners = [
            (cq - 3, cr - 3),
            (cq + 3, cr + 3),
            (cq + 6, cr - 3),
            (cq - 6, cr + 3),
        ]
        if flip:
            corners[0], corners[1] = corners[1], corners[0]
            corners[2], corners[3] = corners[3], corners[2]
        for i in range(self.cfg.house_teams):
            name = _TEAM_NAMES[i % len(_TEAM_NAMES)]
            self.add_team(name, f"house:{name}", corners[i % len(corners)])

    # --- queries ---
    def _tank_at(self, q: int, r: int) -> Tank | None:
        for t in self.tanks.values():
            if t.q == q and t.r == r:
                return t
        return None

    def teams_alive(self) -> set[str]:
        return {t.team for t in self.tanks.values()}

    def team_tanks(self, team: str) -> list[Tank]:
        return [t for t in self.tanks.values() if t.team == team]

    def safe_radius(self) -> float:
        cfg = self.cfg
        full = float(cfg.width)
        t = self.tick_count
        if t <= cfg.zone_start:
            return full
        if t >= cfg.zone_close:
            return float(cfg.zone_min)
        f = (t - cfg.zone_start) / (cfg.zone_close - cfg.zone_start)
        return full - (full - cfg.zone_min) * f

    def _ray(self, oq: int, orr: int, aim: tuple, max_range: int) -> list:
        # the fixed firing line: through the aimed cell, extended to the power's range.
        # Cells ordered by distance from the muzzle; the muzzle's own hex is excluded.
        dist = max(1, hex_distance(oq, orr, aim[0], aim[1]))
        scale = max_range / dist
        ext = (
            round(oq + (aim[0] - oq) * scale),
            round(orr + (aim[1] - orr) * scale),
        )
        cells = [c for c in hex_line(oq, orr, ext[0], ext[1]) if c != (oq, orr)]
        cells = [c for c in cells if self._in_bounds(c[0], c[1])]
        cells.sort(key=lambda c: hex_distance(oq, orr, c[0], c[1]))
        return [c for c in cells if hex_distance(oq, orr, c[0], c[1]) <= max_range]

    def _blocked(
        self,
        aq: int,
        ar: int,
        bq: int,
        br: int,
        tanks_block: bool = False,
        ignore: frozenset | set = frozenset(),
    ) -> bool:
        occ = {(t.q, t.r) for t in self.tanks.values()} - set(ignore) if tanks_block else None
        for q, r in hex_line(aq, ar, bq, br):
            if (q, r) == (aq, ar) or (q, r) == (bq, br):
                continue
            if (q, r) in self.walls:
                return True
            if occ is not None and (q, r) in occ:
                return True  # a tank in the line of fire eats the shot — friend or foe
        return False

    # --- contract ---
    def perceive(self, tank_id: int) -> dict | None:
        me = self.tanks.get(tank_id)
        if me is None:
            return None
        rng = self.cfg.sensor_range
        visible = []
        for o in self.tanks.values():
            if o.id == me.id:
                continue
            d = hex_distance(me.q, me.r, o.q, o.r)
            if d > rng:
                continue
            if self._blocked(me.q, me.r, o.q, o.r):
                continue  # an obstacle breaks line of sight — no seeing through walls
            ally = o.team == me.team
            entry = {
                "id": o.id,
                "kind": "ally" if ally else "enemy",
                "team": o.team,
                "dq": o.q - me.q,
                "dr": o.r - me.r,
                "dist": d,
                "dir": dir_toward(me.q, me.r, o.q, o.r),
                "clear_shot": not self._blocked(me.q, me.r, o.q, o.r, tanks_block=True),
                "step": [o.step_dq, o.step_dr],
            }
            if ally or d <= rng // 2:
                entry["energy"] = round(o.energy)
            visible.append(entry)
        walls = [
            {"dq": wq - me.q, "dr": wr - me.r}
            for (wq, wr) in self.walls
            if hex_distance(me.q, me.r, wq, wr) <= rng
        ]
        cq, cr = self.cfg.width // 2, self.cfg.height // 2
        rad = self.safe_radius()
        return {
            "id": me.id,
            "q": me.q,
            "r": me.r,
            "tick": self.tick_count,
            "width": self.cfg.width,
            "height": self.cfg.height,
            "heading": me.heading,
            "energy": round(me.energy),
            "gun_ready": me.cooldown == 0,
            "hit_taken": round(me.hit_taken),
            "last_fire": me.last_fire,
            "hit_from": me.hit_from,
            "safe": hex_distance(me.q, me.r, cq, cr) <= rad,
            "zone_radius": round(rad, 2),
            "fire_range": self.cfg.fire_range,
            "power_range": list(self.cfg.power_range),
            "power_cost": list(self.cfg.power_cost),
            "map_radius": self.cfg.map_radius,
            "to_center": dir_toward(me.q, me.r, cq, cr),
            "dist_center": hex_distance(me.q, me.r, cq, cr),
            "visible": visible,
            "walls": walls,
        }

    def submit(self, tank_id: int, intent: dict) -> bool:
        if tank_id not in self.tanks:
            return False
        self.pending[tank_id] = intent
        return True

    # --- referee ---
    def step(self) -> dict:
        cfg = self.cfg
        self.tracers = []
        living = list(self.tanks.values())
        self.rng.shuffle(living)
        for t in living:
            t.hit_taken = 0.0
            t.hit_from = 0

        # 1. turns
        for t in living:
            turn = self.pending.get(t.id, {}).get("turn", 0)
            if turn in (-1, 1):
                t.heading = (t.heading + turn) % 6

        # muzzle snapshot: shots are squeezed off from where a tank STARTS the turn, then it
        # displaces — shoot-and-scoot. Targets resolve where they END: breaking line of sight
        # or range with your move voids the incoming shot (the shooter still pays).
        origins = {t.id: (t.q, t.r) for t in living}

        # 2. moves (resolve dest conflicts; one winner per cell)
        wants: dict[tuple[int, int], list[Tank]] = {}
        for t in living:
            mv = self.pending.get(t.id, {}).get("move", "hold")
            if mv not in _VALID_MOVE or mv == "hold":
                continue
            dq, dr = AXIAL_DIRS[t.heading]
            if mv == "back":
                dq, dr = -dq, -dr
            nq, nr = t.q + dq, t.r + dr
            if not self._in_bounds(nq, nr) or (nq, nr) in self.walls:
                continue
            wants.setdefault((nq, nr), []).append(t)
        occupied = {(t.q, t.r) for t in living}
        for (nq, nr), claimants in wants.items():
            if (nq, nr) in occupied:
                continue
            winner = self.rng.choice(claimants)
            occupied.discard((winner.q, winner.r))
            winner.q, winner.r = nq, nr
            occupied.add((nq, nr))
            winner.energy -= cfg.cost_move

        # 3. fire — BALLISTIC. The shooter aims at a CELL: where its chosen enemy stood when
        # the trigger was pulled (turn start), or an explicit fire_at cell for a predicted
        # lead. The shot travels that fixed ray from the shooter's start hex out to the
        # power's range and hits the FIRST thing standing on it after movement resolves —
        # enemy, teammate, or wall. A target that stepped off the ray is MISSED. Every shot
        # has a cause-and-effect line on the board; nothing tracks, nothing homes.
        hit_by: dict[int, Tank] = {}  # who landed the (last) hit, for kill attribution
        for t in living:
            t.last_fire = ""
            intent = self.pending.get(t.id, {})
            tgt_id = intent.get("fire", 0) or 0
            try:
                tgt_id = int(tgt_id)
            except (TypeError, ValueError):
                tgt_id = 0
            fire_at = intent.get("fire_at")
            aim: tuple[int, int] | None = None
            if (
                isinstance(fire_at, (list, tuple))
                and len(fire_at) == 2
                and all(isinstance(c, (int, float)) for c in fire_at)
            ):
                aim = (int(fire_at[0]), int(fire_at[1]))
            try:
                power = int(intent.get("power", 2) or 2)
            except (TypeError, ValueError):
                power = 2
            power = max(1, min(len(cfg.power_range), power))
            cost = cfg.power_cost[power - 1]
            if (not tgt_id and aim is None) or t.cooldown > 0 or t.energy <= 0:
                continue
            # a live tank can ALWAYS pull the trigger — but the gun drains its own energy,
            # and a shot that leaves it at 0 destroys it. Desperation is a choice, not a gate.
            # Pulling the trigger ALWAYS costs energy (scaled by power) and starts the reload.
            t.energy -= cost
            t.cooldown = cfg.gun_cooldown
            oq, orr = origins.get(t.id, (t.q, t.r))
            if aim is None:
                target = self.tanks.get(tgt_id)
                if target is None or target.id == t.id:
                    continue
                aim = origins.get(target.id, (target.q, target.r))
            if aim == (oq, orr):
                t.last_fire = "misfire (aimed at own hex)"
                continue
            self._stat(t.team, "shots")
            self._stat(t.team, "trigger_energy", cost)
            ray = self._ray(oq, orr, aim, cfg.power_range[power - 1])
            occ_now = {(o.q, o.r): o for o in living if o.id != t.id and o.energy > 0}
            victim: Tank | None = None
            stop: tuple[int, int] | None = None
            for cell in ray:
                if cell in self.walls:
                    stop = cell
                    break
                hit = occ_now.get(cell)
                if hit is not None:
                    victim = hit
                    break
            if victim is None:
                t.target = 0
                t.last_fire = "hit cover" if stop else "MISSED — nothing on the line"
                self._stat(t.team, "shots_into_cover" if stop else "shots_missed")
                self.tracers.append(
                    {
                        "q": oq,
                        "r": orr,
                        "tq": (stop or ray[-1])[0],
                        "tr": (stop or ray[-1])[1],
                        "by": t.id,
                        "tgid": 0,
                        "team": t.team,
                        "path": [(oq, orr), *ray[: ray.index(stop) + 1 if stop else len(ray)]],
                        "dmg": 0,
                        "reward": 0,
                        "power": power,
                        "kind": "wall" if stop else "miss",
                    }
                )
                continue
            t.target = victim.id  # the turret aims here (360°); the hull keeps its facing
            victim.energy -= cfg.shot_damage
            victim.hit_taken += cfg.shot_damage
            victim.hit_from = t.id
            hit_by[victim.id] = t
            if victim.team != t.team:
                t.energy = min(cfg.max_energy, t.energy + cfg.hit_reward)
                t.last_fire = f"hit #{victim.id}"
                self._stat(t.team, "shots_hit")
            else:
                t.last_fire = f"HIT TEAMMATE #{victim.id} — they were on your line"
                self._stat(t.team, "teammates_hit")
            self.tracers.append(
                {
                    "q": oq,
                    "r": orr,
                    "tq": victim.q,
                    "tr": victim.r,
                    "by": t.id,
                    "tgid": victim.id,
                    "team": t.team,
                    "path": [(oq, orr), *ray[: ray.index((victim.q, victim.r)) + 1]],
                    "dmg": cfg.shot_damage,
                    "reward": cfg.hit_reward if victim.team != t.team else 0,
                    "power": power,
                    "kind": "hit",
                }
            )

        # 4. cooldown + the closing zone (outside the safe radius = bleed energy) + REPAIR:
        # a tank that held still, didn't pull the trigger, and took no hit recovers a little —
        # so a wounded tank that disengages behind its teammates' screen comes back into the
        # fight instead of being functionally dead. Pressure denies repair; the zone ends camping.
        cq, cr = cfg.width // 2, cfg.height // 2
        rad = self.safe_radius()
        moved_ids = {t.id for t in living if origins.get(t.id) != (t.q, t.r)}
        for t in living:
            oq_, or_ = origins.get(t.id, (t.q, t.r))
            t.step_dq, t.step_dr = t.q - oq_, t.r - or_  # its last move, visible to others
        fired_ids = {t.id for t in living if t.last_fire}
        for t in self.tanks.values():
            if t.cooldown > 0:
                t.cooldown -= 1
            if hex_distance(t.q, t.r, cq, cr) > rad:
                t.energy -= cfg.zone_damage
                self._stat(t.team, "zone_bleed", cfg.zone_damage)
            elif (
                t.id not in moved_ids
                and t.id not in fired_ids
                and t.hit_taken == 0
                and 0 < t.energy < cfg.max_energy
            ):
                gained = min(cfg.max_energy, t.energy + cfg.repair) - t.energy
                t.energy += gained
                self._stat(t.team, "repaired", gained)

        # 5. deaths — logged with attribution AND how far the nearest ally stood, so
        # after-action lessons can measure cohesion failures instead of guessing at them
        dead = [t.id for t in self.tanks.values() if t.energy <= 0]
        for tid in dead:
            t = self.tanks[tid]
            k = hit_by.get(tid)
            if k is not None:
                cause = f"by #{k.id} ({k.team})"
            elif hex_distance(t.q, t.r, cq, cr) > rad:
                cause = "by the closing zone"
            else:
                cause = "burned out — its own shot drained the last energy"
            allies = [o for o in self.tanks.values() if o.team == t.team and o.id != tid]
            near = min((hex_distance(t.q, t.r, o.q, o.r) for o in allies), default=None)
            ally = (
                f", nearest ally {near} hexes away"
                if near is not None
                else ", last tank of its team"
            )
            self.events.append(
                f"tick {self.tick_count}: tank #{tid} ({t.team}) destroyed {cause} "
                f"at ({t.q},{t.r}){ally}"
            )
            del self.tanks[tid]
        del self.events[:-40]

        self.pending.clear()
        self.tick_count += 1
        alive_teams = self.teams_alive()
        if len(alive_teams) == 1:
            self.winner = next(iter(alive_teams))
        elif len(alive_teams) == 0:
            # mutual wipeout this step — nobody outlived the volley: a draw, not a win
            # for whoever happened to be ahead on energy when both shells were in the air
            self.draw = True
        if self.winner is None and self.tick_count >= cfg.match_max_ticks:
            # hard cap: never let a match stall — the team with the most tanks (then energy) wins
            standing: dict[str, tuple[int, float]] = {}
            for t in self.tanks.values():
                cnt, en = standing.get(t.team, (0, 0.0))
                standing[t.team] = (cnt + 1, en + t.energy)
            if standing:
                self.winner = max(standing, key=lambda k: standing[k])
        return self.stats()

    def _stat(self, team: str, key: str, n: float = 1) -> None:
        d = self.match_stats.setdefault(team, {})
        d[key] = d.get(key, 0) + n

    def stats_summary(self) -> str:
        # the after-action facts beyond the kill log: accuracy, friendly fire, cover wasted
        # shots, energy economics, zone discipline — each one a lesson family
        parts = []
        for team, s in sorted(self.match_stats.items()):
            shots = int(s.get("shots", 0))
            parts.append(
                f"{team}: {shots} shots ({int(s.get('shots_hit', 0))} hit, "
                f"{int(s.get('shots_missed', 0))} missed, "
                f"{int(s.get('shots_into_cover', 0))} absorbed by cover, "
                f"{int(s.get('teammates_hit', 0))} hit a TEAMMATE), "
                f"spent {int(s.get('trigger_energy', 0))} energy firing, "
                f"repaired {int(s.get('repaired', 0))} standing still, "
                f"bled {int(s.get('zone_bleed', 0))} to the zone"
            )
        return "; ".join(parts)

    def stats(self) -> dict:
        alive = self.teams_alive()
        return {
            "tick": self.tick_count,
            "teams": len(alive),
            "tanks": len(self.tanks),
            "shots": len(self.tracers),
            "winner": self.winner,
            "draw": self.draw,
            "team_counts": {team: len(self.team_tanks(team)) for team in alive},
        }
