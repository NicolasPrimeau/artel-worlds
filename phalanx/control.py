from __future__ import annotations

from .config import Config
from .tank import dir_toward, hex_distance


def _step_toward(cur: int, target: int) -> int:
    """One turn step (-1/0/1) from heading cur toward target (of 6)."""
    if cur == target:
        return 0
    return 1 if (target - cur) % 6 <= 3 else -1


KNOWLEDGE_TTL = 16  # ticks an enemy sighting stays trusted before it goes stale
LOW_ENERGY = 30.0  # below this a tank kites at max range instead of pressing in


class Bot:
    """A stateful tank controller with memory. Every bot remembers where it last saw
    each enemy and commits to finishing the weakest one it knows of — it never idles,
    never camps. The ONLY difference between the two sides is the board it reads from:
    every Artel tank shares one team board AND concentrates on the single weakest enemy
    the team knows of — focus fire: three guns finish one tank, then the next, so Artel
    snowballs a numbers lead. A Red tank reads only its own board and hunts the NEAREST
    enemy it personally sees — individual seek-and-destroy, fire spread across whoever's
    closest, trades taken one for one. Same engine, same aim; the win comes entirely
    from coordinating the target, which is exactly the Artel edge the demo is selling."""

    def __init__(self, tank_id: int, team: str, board: dict, coord: bool):
        self.id = tank_id
        self.team = team
        self.board = board  # eid -> {q, r, energy, seen}; shared for Artel, private for Red
        self.coord = coord  # True: focus the weakest team-wide. False: chase the nearest.

    def decide(self, p: dict, cfg: Config, tick: int) -> dict:
        # 1. fold everything I see this tick into the board (shared knowledge for Artel)
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

        # 3. outside the closing zone: get back to center before it bleeds you out
        cq, cr = cfg.width // 2, cfg.height // 2
        if not p.get("safe", True):
            cdir = dir_toward(p["q"], p["r"], cq, cr)
            return {"turn": _step_toward(p["heading"], cdir), "move": "fwd"}

        # 4. no enemy known at all: drive to the middle and sweep — never sit still
        if not self.board:
            if hex_distance(p["q"], p["r"], cq, cr) > 2:
                cdir = dir_toward(p["q"], p["r"], cq, cr)
                return {"turn": _step_toward(p["heading"], cdir), "move": "fwd"}
            return {"turn": 1, "move": "fwd"}

        # 5. pick the MOVE target. Artel shares one board and orders by ENERGY first, so
        #    every Artel tank drives toward the same weakest enemy — the team collapses on
        #    one tank at a time. Red reads only its own board and orders by DISTANCE first —
        #    each chases the nearest threat, so the team disperses. That is the whole edge.
        def _key(e: int) -> tuple:
            d = hex_distance(p["q"], p["r"], self.board[e]["q"], self.board[e]["r"])
            energy = self.board[e]["energy"]
            return (energy, d) if self.coord else (d, energy)

        focus = min(self.board, key=_key)
        rec = self.board[focus]

        # FIRE is decided separately from movement so a loaded gun is never wasted: shoot
        # the best enemy in range RIGHT NOW (weakest for Artel, nearest for Red). Every
        # Artel tank that can see the team's weakest target shoots it — focus fire — and
        # any that can't still spend their shot on whatever they can hit.
        intent: dict = {}
        in_range = [e for e in visible if visible[e]["dist"] <= cfg.fire_range]
        if p["gun_ready"] and in_range:
            shot = min(
                in_range,
                key=lambda e: (
                    (self.board[e]["energy"], visible[e]["dist"])
                    if self.coord
                    else (visible[e]["dist"], self.board[e]["energy"])
                ),
            )
            intent["fire"] = shot

        if focus in visible:
            tdir, d, seen = visible[focus]["dir"], visible[focus]["dist"], True
        else:
            tdir = dir_toward(p["q"], p["r"], rec["q"], rec["r"])
            d = hex_distance(p["q"], p["r"], rec["q"], rec["r"])
            seen = False

        if not seen:
            # can't see the team's target: drive to its last-known spot (Artel converges
            # there together); if it's empty on arrival, drop the stale sighting
            intent["turn"] = _step_toward(p["heading"], tdir)
            intent["move"] = "fwd"
            if d <= 1:
                self.board.pop(focus, None)
            return intent

        band = cfg.fire_range - 1 if p["energy"] <= LOW_ENERGY else 2
        if d > band + 1:
            intent["turn"] = _step_toward(p["heading"], tdir)
            intent["move"] = "fwd"  # close the distance
        elif d < band:
            intent["turn"] = _step_toward(p["heading"], tdir)
            intent["move"] = "back"  # too close, ease off (still facing the enemy)
        else:
            # at range: orbit rather than park — keeps moving and swings the angle to
            # break line-of-sight deadlocks behind walls. Firing auto-aims, so heading
            # is free to circle.
            intent["turn"] = _step_toward(p["heading"], (tdir + 1) % 6)
            intent["move"] = "fwd"
        return intent
