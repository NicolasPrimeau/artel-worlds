from __future__ import annotations

import random
from dataclasses import dataclass, field

from .config import Config
from .genome import Genome, random_genome

AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


@dataclass
class Organism:
    id: int
    energy: int
    age: int
    genome: Genome
    lineage_id: int
    agent_id: str | None = None  # set when an external (BYO) agent owns this organism


@dataclass
class Cell:
    q: int
    r: int
    nutrient: int
    toxin: int
    organism: Organism | None = None


class World:
    def __init__(self, cfg: Config, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.tick_count = 0
        self._next_org_id = 0
        self._next_lineage = 0
        self.cells: dict[tuple[int, int], Cell] = {}
        for q in range(cfg.width):
            for r in range(cfg.height):
                self.cells[(q, r)] = Cell(q, r, cfg.nutrient_initial, 0)
        self.organisms: dict[int, Organism] = {}
        # Intention buffer — the shared contract. Local heuristic agents and
        # remote LLM/BYO agents both write here; the tick resolves it.
        self.pending: dict[int, tuple[str, str]] = {}

    # --- topology ---
    def wrap(self, q: int, r: int) -> tuple[int, int] | None:
        if self.cfg.toric:
            return (q % self.cfg.width, r % self.cfg.height)
        if 0 <= q < self.cfg.width and 0 <= r < self.cfg.height:
            return (q, r)
        return None

    def neighbors(self, q: int, r: int) -> list[Cell]:
        out = []
        for dq, dr in AXIAL_DIRS:
            key = self.wrap(q + dq, r + dr)
            if key is not None:
                out.append(self.cells[key])
        return out

    # --- lifecycle ---
    def new_lineage(self) -> int:
        self._next_lineage += 1
        return self._next_lineage

    def spawn(self, q: int, r: int, genome: Genome, lineage_id: int,
              energy: int, agent_id: str | None = None) -> Organism | None:
        cell = self.cells.get((q, r))
        if cell is None or cell.organism is not None:
            return None
        self._next_org_id += 1
        org = Organism(self._next_org_id, energy, 0, genome, lineage_id, agent_id)
        cell.organism = org
        self.organisms[org.id] = org
        return org

    def cell_of(self, org: Organism) -> Cell:
        return self._org_cells()[org.id]

    # --- agent contract (identical for in-process CA and remote LLM/BYO agents) ---
    def perceive(self, org_id: int) -> dict[str, int] | None:
        """Local view an organism gets to decide on. No global state (spec §3b)."""
        cell = self._org_cells().get(org_id)
        if cell is None:
            return None
        org = cell.organism
        neigh = self.neighbors(cell.q, cell.r)
        return {
            "my_energy": org.energy,
            "my_age": org.age,
            "nutrient_here": cell.nutrient,
            "toxin_here": cell.toxin,
            "live_neighbors": sum(1 for n in neigh if n.organism is not None),
            "nutrient_neighbor_max": max((n.nutrient for n in neigh), default=0),
            "toxin_neighbor_max": max((n.toxin for n in neigh), default=0),
            "free_cells": sum(1 for n in neigh if n.organism is None),
        }

    def submit(self, org_id: int, verb: str, target: str) -> bool:
        """Buffer an intention. The tick adjudicates legality + physics (referee)."""
        if org_id not in self.organisms:
            return False
        self.pending[org_id] = (verb, target)
        return True

    def _org_cells(self) -> dict[int, Cell]:
        if not hasattr(self, "_oc") or self._oc_tick != self.tick_count:
            self._oc = {c.organism.id: c for c in self.cells.values() if c.organism}
            self._oc_tick = self.tick_count
        return self._oc

    def seed(self, n: int):
        empties = [c for c in self.cells.values() if c.organism is None]
        self.rng.shuffle(empties)
        for cell in empties[:n]:
            g = random_genome(self.rng, self.cfg.max_genes)
            self.spawn(cell.q, cell.r, g, self.new_lineage(), self.cfg.birth_energy)

    def stats(self) -> dict:
        orgs = list(self.organisms.values())
        pop = len(orgs)
        lineages = len({o.lineage_id for o in orgs})
        avg_energy = sum(o.energy for o in orgs) / pop if pop else 0
        avg_toxin = sum(c.toxin for c in self.cells.values()) / len(self.cells)
        return {
            "tick": self.tick_count,
            "population": pop,
            "lineages": lineages,
            "avg_energy": round(avg_energy, 1),
            "avg_toxin": round(avg_toxin, 1),
        }
