from __future__ import annotations

from collections import defaultdict

from . import physics
from .agent import Agent, HeuristicAgent
from .world import World

DEFAULT_INTENTION = ("metabolize", "random")


def step(world: World, local_agent: Agent | None = None) -> dict:
    """One tick. House organisms (agent_id is None) decide in-process via the
    local agent and submit; remote organisms (agent_id set) have already
    submitted via the same world.submit() surface over HTTP. Both land in
    world.pending and resolve identically."""
    local_agent = local_agent or HeuristicAgent()
    cells = world._org_cells()
    living = list(world.organisms.values())
    world.rng.shuffle(living)

    # --- 1. INTENTIONS ---
    for org in living:
        if org.agent_id is None:  # house/CA organism: decide now, same contract
            perception = world.perceive(org.id)
            if perception is None:
                continue
            verb, target = local_agent.act(perception, org.genome)
            world.submit(org.id, verb, target)
        # remote organisms already wrote world.pending via submit(); missing -> default

    # --- 2. RESOLUTION (referee) ---
    # Build (org, cell, verb, dest) from the buffer; unsubmitted -> default action.
    resolved = []
    for org in living:
        cell = cells.get(org.id)
        if cell is None:
            continue
        verb, target = world.pending.get(org.id, DEFAULT_INTENTION)
        if verb not in ("metabolize", "divide", "migrate", "dormant"):
            verb, target = DEFAULT_INTENTION
        dest = None
        if verb in ("divide", "migrate"):
            dest = physics.select_target(world, cell, target)
            if dest is None:
                verb = "metabolize"
        resolved.append([org, cell, verb, dest])

    # destination conflicts -> one random winner, losers fall back to metabolize
    by_dest: dict[tuple[int, int], list] = defaultdict(list)
    for it in resolved:
        if it[3] is not None:
            by_dest[(it[3].q, it[3].r)].append(it)
    for claimants in by_dest.values():
        if len(claimants) > 1:
            winner = world.rng.choice(claimants)
            for it in claimants:
                if it is not winner:
                    it[2], it[3] = "metabolize", None

    dormant_ids: set[int] = set()
    for org, cell, verb, dest in resolved:
        if verb == "metabolize":
            physics.metabolize(world, org, cell)
        elif verb == "dormant":
            dormant_ids.add(org.id)
        elif verb == "migrate" and dest is not None:
            if not physics.migrate(world, org, cell, dest):
                physics.metabolize(world, org, cell)
        elif verb == "divide" and dest is not None:
            if physics.divide(world, org, cell, dest) is None:
                physics.metabolize(world, org, cell)
        org.age += 1

    world.pending.clear()

    # --- 3. ENVIRONMENT ---
    physics.environment_step(world)
    # --- 4. DEATH ---
    deaths = physics.death_step(world, dormant_ids)
    # --- 5. TRACE ---
    world.tick_count += 1
    s = world.stats()
    s["deaths"] = deaths
    return s
