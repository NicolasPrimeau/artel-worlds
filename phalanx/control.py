from __future__ import annotations

from .config import Config
from .tank import AXIAL_DIRS, dir_toward, hex_distance

KNOWLEDGE_TTL = 16  # ticks an enemy sighting stays trusted before it goes stale
LOW_ENERGY = 30.0  # below this, fall back and keep firing instead of pressing in
ENGAGE_RANGE = 4  # close to here, then HOLD and fire — inside fire_range with a margin for LOS
ZONE_MARGIN = 1  # stay this many hexes inside the safe radius, off the bleeding edge


def _step_toward(cur: int, target: int) -> int:
    """One turn step (-1/0/1) from heading cur toward target (of 6)."""
    if cur == target:
        return 0
    return 1 if (target - cur) % 6 <= 3 else -1


class Bot:
    """One tank's deterministic controller — solo seek-and-destroy with private memory. It
    remembers where it last saw each enemy, drives to a firing position, holds and shoots the
    nearest target it can hit, and retreats from the closing zone. Every tank runs its own Bot
    with its OWN board: deterministic tanks NEVER share what they see. The only coordination in
    Phalanx happens through Artel, among the LLM-driven team — it is never hardcoded here. That
    keeps the demo honest and keeps deterministic-vs-deterministic a fair, unbiased baseline."""

    def __init__(self, tank_id: int, team: str):
        self.id = tank_id
        self.team = team
        self.board: dict[int, dict] = {}  # eid -> {q, r, energy, seen}; private to this tank

    def _toward(self, p: dict, wallset: set, tq: int, tr: int) -> dict:
        """A wall-smoothed step toward (tq, tr): aim at it, but if the cell ahead is cover,
        deflect to the nearest open direction so we slide along walls instead of nosing in."""
        want = dir_toward(p["q"], p["r"], tq, tr)
        for off in (0, 1, -1, 2, -2, 3):
            d = (want + off) % 6
            dq, dr = AXIAL_DIRS[d]
            if (p["q"] + dq, p["r"] + dr) not in wallset:
                return {"turn": _step_toward(p["heading"], d), "move": "fwd"}
        return {"turn": _step_toward(p["heading"], want), "move": "fwd"}

    def decide(self, p: dict, cfg: Config, tick: int) -> dict:
        cq, cr = cfg.width // 2, cfg.height // 2
        wallset = {(p["q"] + w["dq"], p["r"] + w["dr"]) for w in p.get("walls", [])}

        # 1. fold what I personally see this tick into my private board
        visible: dict[int, dict] = {}
        for v in p["visible"]:
            if v["kind"] != "enemy":
                continue
            eq, er = p["q"] + v["dq"], p["r"] + v["dr"]
            self.board[v["id"]] = {"q": eq, "r": er, "energy": v.get("energy", 999), "seen": tick}
            visible[v["id"]] = {"dir": v["dir"], "dist": v["dist"]}

        # 2. forget sightings that have gone stale
        for eid in [e for e, rec in self.board.items() if tick - rec["seen"] > KNOWLEDGE_TTL]:
            del self.board[eid]

        # 3. FIRE, decided independently of movement so a loaded gun is never wasted: shoot
        #    the nearest enemy in range with line of sight.
        intent: dict = {}
        in_range = [e for e in visible if visible[e]["dist"] <= cfg.fire_range]
        if p["gun_ready"] and in_range:
            intent["fire"] = min(in_range, key=lambda e: visible[e]["dist"])

        # 4. the closing zone bleeds energy outside the safe radius — get back inside it,
        #    keeping a margin so we don't loiter on the bleeding edge. No trade is worth
        #    dying to the map.
        zr = p.get("zone_radius", float(cfg.width))
        if hex_distance(p["q"], p["r"], cq, cr) > zr - ZONE_MARGIN:
            return {**intent, **self._toward(p, wallset, cq, cr)}

        # 5. nobody known: push through the center to the mirror cell so the teams cross and
        #    make contact instead of milling about on their own half.
        if not self.board:
            return {
                **intent,
                **self._toward(p, wallset, cfg.width - 1 - p["q"], cfg.height - 1 - p["r"]),
            }

        # 6. press the nearest known enemy
        focus = min(
            self.board,
            key=lambda e: hex_distance(p["q"], p["r"], self.board[e]["q"], self.board[e]["r"]),
        )
        rec = self.board[focus]
        seen = focus in visible

        # hurt: fall back directly away from it while the gun keeps working
        if seen and p["energy"] <= LOW_ENERGY:
            return {
                **intent,
                **self._toward(p, wallset, 2 * p["q"] - rec["q"], 2 * p["r"] - rec["r"]),
            }

        # in range with line of sight: face it and HOLD — a parked tank that can see its
        # target shoots it every cooldown, which is how fights actually resolve.
        if seen and visible[focus]["dist"] <= ENGAGE_RANGE:
            return {
                **intent,
                "turn": _step_toward(p["heading"], visible[focus]["dir"]),
                "move": "hold",
            }

        # otherwise close on its last-known cell to gain range and line of sight
        step = self._toward(p, wallset, rec["q"], rec["r"])
        if not seen and hex_distance(p["q"], p["r"], rec["q"], rec["r"]) <= 1:
            self.board.pop(focus, None)  # arrived at an empty spot — drop the stale sighting
        return {**intent, **step}
