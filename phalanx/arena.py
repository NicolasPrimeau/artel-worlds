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
        self.team_kind: dict[str, str] = {}  # team -> "house:<name>" | "player:<agent>"
        self.pending: dict[int, dict] = {}
        self.walls: set[tuple[int, int]] = set()
        self.winner: str | None = None
        self._scatter_walls()

    # --- setup ---
    def _free_connected(self) -> bool:
        free = {
            (q, r)
            for q in range(self.cfg.width)
            for r in range(self.cfg.height)
            if (q, r) not in self.walls
        }
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
        target = int(self.cfg.width * self.cfg.height * self.cfg.obstacle_density)
        attempts = 0
        while len(self.walls) < target and attempts < target * 25:
            attempts += 1
            q = self.rng.randint(2, self.cfg.width - 3)
            r = self.rng.randint(2, self.cfg.height - 3)
            if (q, r) in self.walls:
                continue
            self.walls.add((q, r))
            if not self._free_connected():
                self.walls.discard((q, r))  # would wall off part of the map — reject

    def _in_bounds(self, q: int, r: int) -> bool:
        return 0 <= q < self.cfg.width and 0 <= r < self.cfg.height

    def _free_cell(self, near: tuple[int, int] | None = None) -> tuple[int, int]:
        for _ in range(200):
            if near is not None:
                q = min(self.cfg.width - 1, max(0, near[0] + self.rng.randint(-3, 3)))
                r = min(self.cfg.height - 1, max(0, near[1] + self.rng.randint(-3, 3)))
            else:
                q = self.rng.randint(0, self.cfg.width - 1)
                r = self.rng.randint(0, self.cfg.height - 1)
            if (q, r) not in self.walls and not self._tank_at(q, r):
                return (q, r)
        return (self.rng.randint(0, self.cfg.width - 1), self.rng.randint(0, self.cfg.height - 1))

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
            (cq - 5, cr - 3),
            (cq + 5, cr + 3),
            (cq + 5, cr - 3),
            (cq - 5, cr + 3),
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

    def _blocked(self, aq: int, ar: int, bq: int, br: int) -> bool:
        for q, r in hex_line(aq, ar, bq, br)[1:-1]:
            if (q, r) in self.walls:
                return True
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
        return {
            "id": me.id,
            "q": me.q,
            "r": me.r,
            "heading": me.heading,
            "energy": round(me.energy),
            "gun_ready": me.cooldown == 0,
            "safe": hex_distance(me.q, me.r, cq, cr) <= self.safe_radius(),
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
        for t in living:
            tgt_id = self.pending.get(t.id, {}).get("fire", 0) or 0
            try:
                tgt_id = int(tgt_id)
            except (TypeError, ValueError):
                tgt_id = 0
            if not tgt_id or t.cooldown > 0 or t.energy <= cfg.shot_cost:
                continue
            target = self.tanks.get(tgt_id)
            if target is None or target.id == t.id:
                continue
            if target.team == t.team and not cfg.friendly_fire:
                continue
            if hex_distance(t.q, t.r, target.q, target.r) > cfg.fire_range:
                continue
            if self._blocked(t.q, t.r, target.q, target.r):
                continue
            t.energy -= cfg.shot_cost
            t.cooldown = cfg.gun_cooldown
            t.heading = dir_toward(t.q, t.r, target.q, target.r)
            t.target = target.id
            target.energy -= cfg.shot_damage
            if target.team != t.team:
                t.energy = min(cfg.max_energy, t.energy + cfg.hit_reward)
            self.tracers.append(
                {"q": t.q, "r": t.r, "tq": target.q, "tr": target.r, "team": t.team}
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

        # 5. deaths
        dead = [t.id for t in self.tanks.values() if t.energy <= 0]
        for tid in dead:
            del self.tanks[tid]

        self.pending.clear()
        self.tick_count += 1
        alive_teams = self.teams_alive()
        if len(alive_teams) == 1:
            self.winner = next(iter(alive_teams))
        elif len(alive_teams) == 0 and pre_energy:
            # mutual wipeout this step — no draw: the team that was ahead wins
            self.winner = max(pre_energy, key=lambda k: pre_energy[k])
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
