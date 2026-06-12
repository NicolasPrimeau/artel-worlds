from __future__ import annotations

from .config import Config
from .tank import AXIAL_DIRS, dir_toward, hex_distance

KNOWLEDGE_TTL = 16  # ticks an enemy sighting stays trusted before it goes stale
LOW_ENERGY = 30.0  # below this, fall back and keep firing instead of pressing in
ZONE_MARGIN = 1  # stay this many hexes inside the safe radius, off the bleeding edge

# Three solo temperaments — different comfort range, target pick, and hunting lane. A team of
# deterministic bots fields one of each, so "the solo side" is three DIFFERENT hunters, not one
# policy times three (identical policies marched as an accidental phalanx and collected
# focus-fire nobody coordinated).
#   brawler     — charges to point blank, shoots whoever is closest, takes the middle lane
#   ranger      — holds at max range and kites anything that closes, sweeps the right lane
#   opportunist — mid-range, singles out the WEAKEST enemy it knows of, sweeps the left lane
STRATEGIES = ("brawler", "ranger", "opportunist")
_TRAITS = {
    "brawler": {"min_d": 1, "max_d": 2, "pick": "near", "lane": 0},
    "ranger": {"min_d": 5, "max_d": 6, "pick": "near", "lane": 3},
    "opportunist": {"min_d": 3, "max_d": 4, "pick": "weak", "lane": -3},
}


def _step_toward(cur: int, target: int) -> int:
    """One turn step (-1/0/1) from heading cur toward target (of 6)."""
    if cur == target:
        return 0
    return 1 if (target - cur) % 6 <= 3 else -1


class Bot:
    """One tank's deterministic controller — solo seek-and-destroy with private memory and its
    own temperament. It remembers where it last saw each enemy, drives to its preferred firing
    distance, shoots what its temperament points at, and retreats from the closing zone. Every
    tank runs its own Bot with its OWN board: deterministic tanks NEVER share what they see.
    The only coordination in Phalanx happens through Artel, among the LLM-driven team — it is
    never hardcoded here."""

    def __init__(self, tank_id: int, team: str, strategy: str | None = None):
        self.id = tank_id
        self.team = team
        self.strategy = strategy or STRATEGIES[tank_id % len(STRATEGIES)]
        self.traits = _TRAITS[self.strategy]
        self.board: dict[int, dict] = {}  # eid -> {q, r, energy, seen}; private to this tank

    def _toward(self, p: dict, wallset: set, tq: int, tr: int, R: int) -> dict:
        """A wall-smoothed step toward (tq, tr): aim at it, but if the cell ahead is cover
        or off the hexagon, deflect to the nearest open direction so we slide along edges
        instead of nosing in."""
        want = dir_toward(p["q"], p["r"], tq, tr)
        for off in (0, 1, -1, 2, -2, 3):
            d = (want + off) % 6
            dq, dr = AXIAL_DIRS[d]
            nq, nr = p["q"] + dq, p["r"] + dr
            if (nq, nr) not in wallset and hex_distance(nq, nr, R, R) <= R:
                return {"turn": _step_toward(p["heading"], d), "move": "fwd"}
        return {"turn": _step_toward(p["heading"], want), "move": "fwd"}

    def decide(self, p: dict, cfg: Config, tick: int) -> dict:
        cq, cr = cfg.width // 2, cfg.height // 2
        R = cfg.map_radius
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

        # 3. FIRE, decided independently of movement so a loaded gun is never wasted — at the
        #    enemy this temperament points at: weakest in range for an opportunist, nearest
        #    for everyone else.
        intent: dict = {}
        in_range = [e for e in visible if visible[e]["dist"] <= cfg.fire_range]
        if p["gun_ready"] and in_range:
            if self.traits["pick"] == "weak":
                tgt = min(in_range, key=lambda e: (self.board[e]["energy"], visible[e]["dist"]))
            else:
                tgt = min(in_range, key=lambda e: visible[e]["dist"])
            intent["fire"] = tgt
            d = visible[tgt]["dist"]
            intent["power"] = next(i + 1 for i, rng_ in enumerate(cfg.power_range) if d <= rng_)

        # 4. the closing zone bleeds energy outside the safe radius — get back inside it,
        #    keeping a margin so we don't loiter on the bleeding edge. No trade is worth
        #    dying to the map.
        zr = p.get("zone_radius", float(cfg.width))
        if hex_distance(p["q"], p["r"], cq, cr) > zr - ZONE_MARGIN:
            return {**intent, **self._toward(p, wallset, cq, cr, R)}

        # 5. nobody known: sweep toward the enemy half ALONE, each temperament down its own
        #    lane — solo hunters spread; they don't get to march as an accidental phalanx.
        if not self.board:
            lane = self.traits["lane"]
            tq = max(0, min(cfg.width - 1, cfg.width - 1 - p["q"] + lane))
            tr = max(0, min(cfg.height - 1, cfg.height - 1 - p["r"] - lane))
            return {**intent, **self._toward(p, wallset, tq, tr, R)}

        # 6. work the enemy this temperament points at
        if self.traits["pick"] == "weak":
            focus = min(
                self.board,
                key=lambda e: (
                    self.board[e]["energy"],
                    hex_distance(p["q"], p["r"], self.board[e]["q"], self.board[e]["r"]),
                ),
            )
        else:
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
                **self._toward(p, wallset, 2 * p["q"] - rec["q"], 2 * p["r"] - rec["r"], R),
            }

        if seen:
            d = visible[focus]["dist"]
            if d < self.traits["min_d"]:
                # closer than this temperament likes: open the gap while the gun keeps working
                return {
                    **intent,
                    **self._toward(p, wallset, 2 * p["q"] - rec["q"], 2 * p["r"] - rec["r"], R),
                }
            if d <= self.traits["max_d"]:
                # in the comfort band: face it and HOLD — a parked tank that can see its
                # target shoots it every cooldown, which is how fights actually resolve
                return {
                    **intent,
                    "turn": _step_toward(p["heading"], visible[focus]["dir"]),
                    "move": "hold",
                }

        # otherwise close on its last-known cell to gain range and line of sight
        step = self._toward(p, wallset, rec["q"], rec["r"], R)
        if not seen and hex_distance(p["q"], p["r"], rec["q"], rec["r"]) <= 1:
            self.board.pop(focus, None)  # arrived at an empty spot — drop the stale sighting
        return {**intent, **step}
