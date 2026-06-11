from __future__ import annotations

from .config import Config
from .tank import bearing


def _step_toward(cur: int, target: int) -> int:
    """One turn step (-1/0/1) from heading cur toward target (of 8)."""
    if cur == target:
        return 0
    return 1 if (target - cur) % 8 <= 4 else -1


def decide(p: dict, cfg: Config) -> dict:
    """A tank's default brain: find the nearest enemy, line up the gun, fire when
    aligned, close the distance, and flee when low on energy. Sensors only — this
    is what a tank does when no external agent is steering it."""
    enemies = [v for v in p["visible"] if v["kind"] == "enemy"]
    if not enemies:
        # nothing in sight: head to the middle so teams converge and fight, sweeping the gun
        dx, dy = cfg.width // 2 - p["x"], cfg.height // 2 - p["y"]
        if abs(dx) + abs(dy) > 4:
            cdir = bearing(dx, dy)
            return {"turn": _step_toward(p["heading"], cdir), "move": "fwd", "aim": cdir}
        return {"move": "fwd", "aim": (p["gun_heading"] + 1) % 8}

    target = min(enemies, key=lambda e: e["dist"])
    tdir = target["dir"]
    intent: dict = {"aim": tdir}

    # flee if badly hurt: turn away and run, hold fire
    if p["energy"] < 25:
        away = (tdir + 4) % 8
        intent["turn"] = _step_toward(p["heading"], away)
        intent["move"] = "fwd"
        return intent

    # fire when the gun is lined up and loaded; more power up close
    if p["gun_ready"] and p["gun_heading"] == tdir:
        power = cfg.fire_max if target["dist"] <= 3 else 1.0
        intent["fire"] = max(cfg.fire_min, min(power, p["energy"] * 0.3))

    # face the enemy and manage range
    intent["turn"] = _step_toward(p["heading"], tdir)
    if target["dist"] > 4:
        intent["move"] = "fwd"
    elif target["dist"] < 2:
        intent["move"] = "back"
    return intent
