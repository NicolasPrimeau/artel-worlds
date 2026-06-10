from __future__ import annotations


from .genome import crossover, mutate
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
        return max(
            free, key=lambda c: sum(1 for n in world.neighbors(c.q, c.r) if n.organism is None)
        )
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


def _donor(world: World, cell: Cell) -> Organism | None:
    occupied = [n.organism for n in world.neighbors(cell.q, cell.r) if n.organism is not None]
    if occupied:
        return world.rng.choice(occupied)
    if world.organisms:
        return world.rng.choice(list(world.organisms.values()))
    return None


def divide(world: World, org: Organism, cell: Cell, dest: Cell) -> Organism | None:
    cfg = world.cfg
    if dest.organism is not None or org.energy < cfg.cost_division:
        return None
    org.energy -= cfg.cost_division
    share = org.energy // 2
    org.energy -= share
    base = org.genome
    if cfg.p_crossover and world.rng.random() < cfg.p_crossover:
        donor = _donor(world, cell)
        if donor is not None and donor is not org:
            base = crossover(org.genome, donor.genome, world.rng, cfg.max_genes)
    child_genome = mutate(base, world.rng, cfg)
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


def death_step(world: World) -> int:
    cfg = world.cfg
    dead = []
    for cell in world.cells.values():
        org = cell.organism
        if org is None:
            continue
        if org.energy <= 0 or cell.toxin >= cfg.toxin_lethal or org.age >= cfg.max_age:
            dead.append((cell, org))
    for cell, org in dead:
        cell.organism = None
        world.organisms.pop(org.id, None)
    return len(dead)
