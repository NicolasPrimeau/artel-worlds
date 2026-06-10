from __future__ import annotations

import random

from .config import Config
from .genome import mutate
from .world import Cell, Organism, World


def select_target(world: World, cell: Cell, target: str) -> Cell | None:
    free = [n for n in world.neighbors(cell.q, cell.r) if n.organism is None]
    if not free:
        return None
    if target == "nutrient_max":
        return max(free, key=lambda c: c.nutrient)
    if target == "toxin_min":
        return min(free, key=lambda c: c.toxin)
    if target == "empty_max":
        return max(free, key=lambda c: sum(
            1 for n in world.neighbors(c.q, c.r) if n.organism is None))
    return world.rng.choice(free)


def metabolize(world: World, org: Organism, cell: Cell) -> None:
    cfg = world.cfg
    org.energy -= cfg.cost_base
    consumed = min(cell.nutrient, cfg.consumption_max)
    cell.nutrient -= consumed
    org.energy += cfg.gain_per_nutrient * consumed
    cell.toxin = min(cfg.toxin_max, cell.toxin + cfg.toxin_emission)


def migrate(world: World, org: Organism, cell: Cell, dest: Cell) -> bool:
    if dest.organism is not None:
        return False
    org.energy -= world.cfg.cost_migration
    cell.organism = None
    dest.organism = org
    return True


def divide(world: World, org: Organism, cell: Cell, dest: Cell) -> Organism | None:
    cfg = world.cfg
    if dest.organism is not None or org.energy < cfg.cost_division:
        return None
    org.energy -= cfg.cost_division
    share = org.energy // 2
    org.energy -= share
    child_genome = mutate(org.genome, world.rng, cfg)
    world._next_org_id += 1
    child = Organism(world._next_org_id, share, 0, child_genome, org.lineage_id)
    dest.organism = child
    world.organisms[child.id] = child
    return child


def environment_step(world: World) -> None:
    cfg = world.cfg
    for cell in world.cells.values():
        cell.nutrient = min(cfg.nutrient_max, cell.nutrient + cfg.nutrient_regrowth)
        cell.toxin = max(0, cell.toxin - cfg.toxin_degradation)


def death_step(world: World, dormant_ids: set[int] | None = None) -> int:
    cfg = world.cfg
    dormant_ids = dormant_ids or set()
    dead = []
    for cell in world.cells.values():
        org = cell.organism
        if org is None:
            continue
        starved = org.energy <= 0 and org.id not in dormant_ids
        poisoned = cell.toxin >= cfg.toxin_lethal
        if starved or poisoned:
            dead.append((cell, org))
    for cell, org in dead:
        cell.organism = None
        world.organisms.pop(org.id, None)
    return len(dead)
