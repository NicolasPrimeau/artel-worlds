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
        # team standing going into this step — used to break a mutual wipeout so a
        # match never ends in a draw: whoever was ahead when both fell takes it.
        pre_energy: dict[str, float] = {}
        for t in living:
            pre_energy[t.team] = pre_energy.get(t.team, 0.0) + t.energy

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

        # 3. fire — each tank shoots AT a chosen enemy; a shot lands if the target
        # is in range and there's line of sight. No ray to line up: reliable hits.
        hit_by: dict[int, Tank] = {}  # who landed the (last) hit, for kill attribution
        for t in living:
            tgt_id = self.pending.get(t.id, {}).get("fire", 0) or 0
            try:
                tgt_id = int(tgt_id)
            except (TypeError, ValueError):
                tgt_id = 0
            try:
                power = int(self.pending.get(t.id, {}).get("power", 2) or 2)
            except (TypeError, ValueError):
                power = 2
            power = max(1, min(len(cfg.power_range), power))
            cost = cfg.power_cost[power - 1]
            if not tgt_id or t.cooldown > 0 or t.energy <= 0:
                continue
            # a live tank can ALWAYS pull the trigger — but the gun drains its own energy,
            # and a shot that leaves it at 0 destroys it. Desperation is a choice, not a gate.
            # pulling the trigger ALWAYS costs energy (scaled by power) and starts the
            # reload — a shot at a target out of range or behind cover is wasted, not free
            t.energy -= cost
            t.cooldown = cfg.gun_cooldown
            target = self.tanks.get(tgt_id)
            if target is None or target.id == t.id:
                continue
            if target.team == t.team and not cfg.friendly_fire:
                continue
            oq, orr = origins.get(t.id, (t.q, t.r))
            if hex_distance(oq, orr, target.q, target.r) > cfg.power_range[power - 1]:
                continue
            if self._blocked(oq, orr, target.q, target.r, tanks_block=True, ignore={(t.q, t.r)}):
                continue
            t.target = target.id  # the turret aims here (360°); the hull keeps its facing
            target.energy -= cfg.shot_damage
            target.hit_taken += cfg.shot_damage
            target.hit_from = t.id
            hit_by[target.id] = t
            if target.team != t.team:
                t.energy = min(cfg.max_energy, t.energy + cfg.hit_reward)
            self.tracers.append(
                {
                    "q": oq,
                    "r": orr,
                    "tq": target.q,
                    "tr": target.r,
                    "team": t.team,
                    "path": hex_line(oq, orr, target.q, target.r),
                    "dmg": cfg.shot_damage,
                    "reward": cfg.hit_reward,
                    "power": power,
                }
            )

        # 4. cooldown + the closing zone (outside the safe radius = bleed energy)
        cq, cr = cfg.width // 2, cfg.height // 2
        rad = self.safe_radius()
        for t in self.tanks.values():
            t.energy = min(cfg.max_energy, t.energy + cfg.regen)
            if t.cooldown > 0:
                t.cooldown -= 1
            if hex_distance(t.q, t.r, cq, cr) > rad:
                t.energy -= cfg.zone_damage

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
        elif len(alive_teams) == 0 and pre_energy:
            # mutual wipeout this step — no draw: the team that was ahead wins
            self.winner = max(pre_energy, key=lambda k: pre_energy[k])
        if self.winner is None and self.tick_count >= cfg.match_max_ticks:
            # hard cap: never let a match stall — the team with the most tanks (then energy) wins
            standing: dict[str, tuple[int, float]] = {}
            for t in self.tanks.values():
                cnt, en = standing.get(t.team, (0, 0.0))
                standing[t.team] = (cnt + 1, en + t.energy)
            if standing:
                self.winner = max(standing, key=lambda k: standing[k])
        return self.stats()

    def stats(self) -> dict:
        alive = self.teams_alive()
        return {
            "tick": self.tick_count,
            "teams": len(alive),
            "tanks": len(self.tanks),
            "shots": len(self.tracers),
            "winner": self.winner,
            "team_counts": {team: len(self.team_tanks(team)) for team in alive},
        }
