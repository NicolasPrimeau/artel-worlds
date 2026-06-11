from __future__ import annotations

import random

from .config import Config
from .tank import DIRS, Shell, Tank, bearing, turn_toward

_TEAM_NAMES = ("blue", "red", "green", "gold", "violet", "cyan")
_VALID_MOVE = ("fwd", "back", "hold")


def _cheb(ax: int, ay: int, bx: int, by: int) -> int:
    return max(abs(ax - bx), abs(ay - by))


class Arena:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tick_count = 0
        self._next_id = 0
        self.tanks: dict[int, Tank] = {}
        self.shells: list[Shell] = []
        self.team_kind: dict[str, str] = {}  # team -> "house:<name>" | "player:<agent>"
        self.pending: dict[int, dict] = {}
        self.walls: set[tuple[int, int]] = set()
        self.winner: str | None = None
        self._scatter_walls()

    # --- setup ---
    def _scatter_walls(self) -> None:
        n = int(self.cfg.width * self.cfg.height * self.cfg.obstacle_density)
        for _ in range(n):
            x = self.rng.randint(2, self.cfg.width - 3)
            y = self.rng.randint(2, self.cfg.height - 3)
            self.walls.add((x, y))

    def _free_cell(self, near: tuple[int, int] | None = None) -> tuple[int, int]:
        for _ in range(200):
            if near is not None:
                x = min(self.cfg.width - 1, max(0, near[0] + self.rng.randint(-3, 3)))
                y = min(self.cfg.height - 1, max(0, near[1] + self.rng.randint(-3, 3)))
            else:
                x = self.rng.randint(0, self.cfg.width - 1)
                y = self.rng.randint(0, self.cfg.height - 1)
            if (x, y) not in self.walls and not self._tank_at(x, y):
                return (x, y)
        return (self.rng.randint(0, self.cfg.width - 1), self.rng.randint(0, self.cfg.height - 1))

    def add_team(self, team: str, controller: str, anchor: tuple[int, int]) -> list[int]:
        self.team_kind[team] = controller
        ids = []
        for _ in range(self.cfg.team_size):
            x, y = self._free_cell(near=anchor)
            ids.append(self._spawn(team, controller, x, y))
        return ids

    def _spawn(self, team: str, controller: str, x: int, y: int) -> int:
        self._next_id += 1
        heading = self.rng.randint(0, 7)
        self.tanks[self._next_id] = Tank(
            id=self._next_id,
            team=team,
            x=x,
            y=y,
            heading=heading,
            gun=heading,
            energy=float(self.cfg.start_energy),
            controller=controller,
        )
        return self._next_id

    def seed_house(self) -> None:
        corners = [
            (3, 3),
            (self.cfg.width - 4, self.cfg.height - 4),
            (self.cfg.width - 4, 3),
            (3, self.cfg.height - 4),
        ]
        for i in range(self.cfg.house_teams):
            name = _TEAM_NAMES[i % len(_TEAM_NAMES)]
            self.add_team(name, f"house:{name}", corners[i % len(corners)])

    # --- queries ---
    def _tank_at(self, x: int, y: int) -> Tank | None:
        for t in self.tanks.values():
            if t.x == x and t.y == y:
                return t
        return None

    def teams_alive(self) -> set[str]:
        return {t.team for t in self.tanks.values()}

    def team_tanks(self, team: str) -> list[Tank]:
        return [t for t in self.tanks.values() if t.team == team]

    # --- contract ---
    def perceive(self, tank_id: int) -> dict | None:
        me = self.tanks.get(tank_id)
        if me is None:
            return None
        r = self.cfg.sensor_range
        visible = []
        for o in self.tanks.values():
            if o.id == me.id:
                continue
            d = _cheb(me.x, me.y, o.x, o.y)
            if d > r:
                continue
            dx, dy = o.x - me.x, o.y - me.y
            ally = o.team == me.team
            entry = {
                "id": o.id,
                "kind": "ally" if ally else "enemy",
                "team": o.team,
                "dx": dx,
                "dy": dy,
                "dist": d,
                "dir": bearing(dx, dy),
            }
            if ally or d <= r // 2:  # enemy energy only revealed up close
                entry["energy"] = round(o.energy)
            visible.append(entry)
        walls = [
            {"dx": wx - me.x, "dy": wy - me.y}
            for (wx, wy) in self.walls
            if _cheb(me.x, me.y, wx, wy) <= r
        ]
        return {
            "id": me.id,
            "x": me.x,
            "y": me.y,
            "heading": me.heading,
            "gun_heading": me.gun,
            "energy": round(me.energy),
            "gun_ready": me.cooldown == 0,
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
        living = list(self.tanks.values())
        self.rng.shuffle(living)

        # 1. turns
        for t in living:
            turn = self.pending.get(t.id, {}).get("turn", 0)
            if turn in (-1, 1):
                t.heading = (t.heading + turn) % 8

        # 2. moves (resolve dest conflicts; one winner per cell)
        wants: dict[tuple[int, int], list[Tank]] = {}
        for t in living:
            mv = self.pending.get(t.id, {}).get("move", "hold")
            if mv not in _VALID_MOVE or mv == "hold":
                continue
            dx, dy = DIRS[t.heading]
            if mv == "back":
                dx, dy = -dx, -dy
            nx, ny = t.x + dx, t.y + dy
            if not (0 <= nx < cfg.width and 0 <= ny < cfg.height) or (nx, ny) in self.walls:
                continue
            wants.setdefault((nx, ny), []).append(t)
        occupied = {(t.x, t.y) for t in living}
        for (nx, ny), claimants in wants.items():
            if (nx, ny) in occupied:
                continue  # a stationary tank holds the cell
            winner = self.rng.choice(claimants)
            occupied.discard((winner.x, winner.y))
            winner.x, winner.y = nx, ny
            occupied.add((nx, ny))
            winner.energy -= cfg.cost_move

        # 3. aim
        for t in living:
            aim = self.pending.get(t.id, {}).get("aim")
            if isinstance(aim, int) and 0 <= aim <= 7:
                t.gun = turn_toward(t.gun, aim)

        # 4. fire
        for t in living:
            power = self.pending.get(t.id, {}).get("fire", 0) or 0
            try:
                power = float(power)
            except (TypeError, ValueError):
                power = 0
            if power < cfg.fire_min or t.cooldown > 0 or t.energy <= power:
                continue
            power = min(power, cfg.fire_max)
            t.energy -= power
            t.cooldown = cfg.gun_cooldown
            dx, dy = DIRS[t.gun]
            self.shells.append(Shell(t.x, t.y, dx, dy, power, t.team, t.id))

        # 5. advance shells, one cell at a time (no tunneling)
        survivors = []
        for s in self.shells:
            alive = True
            for _ in range(cfg.shell_speed):
                s.x += s.dx
                s.y += s.dy
                if not (0 <= s.x < cfg.width and 0 <= s.y < cfg.height) or (s.x, s.y) in self.walls:
                    alive = False
                    break
                hit = self._tank_at(s.x, s.y)
                if hit and (cfg.friendly_fire or hit.team != s.team) and hit.id != s.shooter:
                    hit.energy -= cfg.damage_per_power * s.power
                    shooter = self.tanks.get(s.shooter)
                    if shooter and shooter.team != hit.team:
                        shooter.energy = min(
                            cfg.max_energy, shooter.energy + cfg.reward_per_power * s.power
                        )
                    alive = False
                    break
            if alive:
                survivors.append(s)
        self.shells = survivors

        # 6. regen + cooldown
        for t in self.tanks.values():
            t.energy = min(cfg.max_energy, t.energy + cfg.regen)
            if t.cooldown > 0:
                t.cooldown -= 1

        # 7. deaths
        dead = [t.id for t in self.tanks.values() if t.energy <= 0]
        for tid in dead:
            del self.tanks[tid]

        self.pending.clear()
        self.tick_count += 1
        alive_teams = self.teams_alive()
        if len(alive_teams) <= 1:
            self.winner = next(iter(alive_teams), None)
        return self.stats()

    def stats(self) -> dict:
        alive = self.teams_alive()
        return {
            "tick": self.tick_count,
            "teams": len(alive),
            "tanks": len(self.tanks),
            "shells": len(self.shells),
            "winner": self.winner,
            "team_counts": {team: len(self.team_tanks(team)) for team in alive},
        }
