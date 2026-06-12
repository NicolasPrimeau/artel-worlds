from __future__ import annotations

from .config import Config
from .tank import AXIAL_DIRS, bfs_step, dir_toward, hex_distance

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
        # standing ORDERS from a commander (the Artel-coordinated layer). Red bots never
        # receive any — orders are exactly the thing coordination buys.
        #   focus: enemy id to prioritize     focus_at: (q,r) intel seed for that enemy
        #   regroup: (q,r) move there now     post: (q,r) hold there when nothing is known
        self.orders: dict = {}

    def _toward(self, p: dict, walls: set, occ: set, tq: int, tr: int, R: int) -> dict:
        """One step of a BFS path toward (tq, tr) around known walls (tanks veto only the
        immediate step — they move). Falls back to a wall-sliding greedy step when no route
        is known."""
        d = bfs_step(p["q"], p["r"], tq, tr, walls, R, soft=occ)
        if d is None:
            want = dir_toward(p["q"], p["r"], tq, tr)
            for off in (0, 1, -1, 2, -2, 3):
                dd = (want + off) % 6
                dq, dr = AXIAL_DIRS[dd]
                nq, nr = p["q"] + dq, p["r"] + dr
                blocked = walls | occ
                if (nq, nr) not in blocked and hex_distance(nq, nr, R, R) <= R:
                    d = dd
                    break
            else:
                d = want
        return {"turn": _step_toward(p["heading"], d), "move": "fwd"}

    def decide(self, p: dict, cfg: Config, tick: int) -> dict:
        cq, cr = cfg.width // 2, cfg.height // 2
        R = cfg.map_radius
        wallset = {(p["q"] + w["dq"], p["r"] + w["dr"]) for w in p.get("walls", [])}
        # tanks are as solid as cover, but only for the immediate step — they move
        occ = {(p["q"] + v["dq"], p["r"] + v["dr"]) for v in p.get("visible", [])}

        # 0. commander intel: a focus order can carry the target's reported position —
        # a sighting this tank never made itself lands on its board (the Artel edge)
        fa = self.orders.pop("focus_at", None)
        if fa and self.orders.get("focus"):
            prev = self.board.get(self.orders["focus"], {})
            self.board[self.orders["focus"]] = {
                "q": int(fa[0]),
                "r": int(fa[1]),
                "energy": prev.get("energy", 999),
                "seen": tick,
            }

        # 1. fold what I personally see this tick into my private board
        visible: dict[int, dict] = {}
        for v in p["visible"]:
            if v["kind"] != "enemy":
                continue
            eq, er = p["q"] + v["dq"], p["r"] + v["dr"]
            self.board[v["id"]] = {"q": eq, "r": er, "energy": v.get("energy", 999), "seen": tick}
            visible[v["id"]] = {
                "dir": v["dir"],
                "dist": v["dist"],
                "clear": v.get("clear_shot", True),
            }

        # 2. forget sightings that have gone stale
        for eid in [e for e, rec in self.board.items() if tick - rec["seen"] > KNOWLEDGE_TTL]:
            del self.board[eid]

        # 3. FIRE, decided independently of movement so a loaded gun is never wasted — at the
        #    enemy this temperament points at: weakest in range for an opportunist, nearest
        #    for everyone else.
        intent: dict = {}
        in_range = [
            e
            for e in visible
            if visible[e]["dist"] <= cfg.fire_range and visible[e].get("clear", True)
        ]
        if p["gun_ready"] and in_range:
            if self.orders.get("focus") in in_range:
                tgt = self.orders["focus"]  # the called target outranks temperament
            elif self.traits["pick"] == "weak":
                tgt = min(in_range, key=lambda e: (self.board[e]["energy"], visible[e]["dist"]))
            else:
                tgt = min(in_range, key=lambda e: visible[e]["dist"])
            d = visible[tgt]["dist"]
            need = next(i + 1 for i, rng_ in enumerate(cfg.power_range) if d <= rng_)
            # bots stay prudent: fire only when the shot won't drain them to death
            if p["energy"] > cfg.power_cost[need - 1]:
                intent["fire"] = tgt
                intent["power"] = need

        # 4. the closing zone bleeds energy outside the safe radius — get back inside it,
        #    keeping a margin so we don't loiter on the bleeding edge. No trade is worth
        #    dying to the map.
        zr = p.get("zone_radius", float(cfg.width))
        if hex_distance(p["q"], p["r"], cq, cr) > zr - ZONE_MARGIN:
            return {**intent, **self._toward(p, wallset, occ, cq, cr, R)}

        # 4a. REPAIR: hurt, nothing in sight, inside the zone — park and recover (+2/turn
        # while idle, unhit, and not firing). Both teams run this same self-preservation;
        # the smarter USE of it (rotations, screens) is the commander's job.
        if (
            p["energy"] <= LOW_ENERGY
            and p.get("safe", True)
            and not any(v["kind"] == "enemy" for v in p.get("visible", []))
            and not self.orders.get("regroup")
        ):
            return {"turn": 0, "move": "hold"}

        # 4b. REGROUP order: the commander called a rally — go there now (the gun keeps
        # working on the way); the order clears itself on arrival
        rg = self.orders.get("regroup")
        if rg:
            if hex_distance(p["q"], p["r"], rg[0], rg[1]) <= 1:
                self.orders.pop("regroup", None)
            else:
                return {**intent, **self._toward(p, wallset, occ, rg[0], rg[1], R)}

        # 5. nobody known: sweep toward the enemy half ALONE, each temperament down its own
        #    lane — solo hunters spread; they don't get to march as an accidental phalanx.
        #    Lane targets get pulled back onto the hexagon (the bounding box's corners are
        #    not part of the map).
        if not self.board:
            post = self.orders.get("post")
            if post:
                if hex_distance(p["q"], p["r"], post[0], post[1]) > 1:
                    return {**intent, **self._toward(p, wallset, occ, post[0], post[1], R)}
                return {**intent, "turn": 0, "move": "hold"}  # holding the assigned post
            lane = self.traits["lane"]
            tq = 2 * cq - p["q"] + lane
            tr = 2 * cr - p["r"] - lane
            while hex_distance(tq, tr, cq, cr) > R:
                tq += 1 if tq < cq else -1 if tq > cq else 0
                tr += 1 if tr < cr else -1 if tr > cr else 0
            return {**intent, **self._toward(p, wallset, occ, tq, tr, R)}

        # 6. work the enemy this temperament points at — unless the commander called one
        if self.orders.get("focus") in self.board:
            focus = self.orders["focus"]
        elif self.traits["pick"] == "weak":
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
                **self._toward(p, wallset, occ, 2 * p["q"] - rec["q"], 2 * p["r"] - rec["r"], R),
            }

        if seen:
            d = visible[focus]["dist"]
            if d < self.traits["min_d"]:
                # closer than this temperament likes: open the gap while the gun keeps working
                return {
                    **intent,
                    **self._toward(
                        p, wallset, occ, 2 * p["q"] - rec["q"], 2 * p["r"] - rec["r"], R
                    ),
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
        step = self._toward(p, wallset, occ, rec["q"], rec["r"], R)
        if not seen and hex_distance(p["q"], p["r"], rec["q"], rec["r"]) <= 1:
            self.board.pop(focus, None)  # arrived at an empty spot — drop the stale sighting
        return {**intent, **step}
