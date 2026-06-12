from __future__ import annotations

import math

import random
from dataclasses import dataclass

from .config import Config
from .genome import Genome, random_genome

AXIAL_DIRS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
HOUSE_NAMES = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
)


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
        # A tribe = a lineage_id + a controller. "house:<name>" tribes are driven
        # in-process; a player tribe's controller is the joining agent_id; wild
        # immigrant lineages are unregistered (also driven in-process).
        self.tribes: dict[int, str] = {}
        # House lineages whose decisions come from an LLM (the rest use the CA).
        self.llm_tribes: set[int] = set()
        # Intention buffer — the shared contract. In-process agents and remote
        # players both write here; the tick resolves it.
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

    # --- tribes ---
    def register_tribe(self, lineage_id: int, controller: str) -> None:
        self.tribes[lineage_id] = controller

    def controller_of(self, lineage_id: int) -> str | None:
        return self.tribes.get(lineage_id)

    def is_player_tribe(self, lineage_id: int) -> bool:
        c = self.tribes.get(lineage_id)
        return bool(c) and not c.startswith("house:")

    def tribe_members(self, lineage_id: int) -> list[Organism]:
        return [o for o in self.organisms.values() if o.lineage_id == lineage_id]

    # --- lifecycle ---
    def new_lineage(self) -> int:
        self._next_lineage += 1
        return self._next_lineage

    def spawn(
        self,
        q: int,
        r: int,
        genome: Genome,
        lineage_id: int,
        energy: int,
        agent_id: str | None = None,
    ) -> Organism | None:
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
        # Seed n organisms split across house_tribes lineages — each tribe CLUSTERED around
        # its own settlement, settlements spread across the map. Tribes start as villages,
        # not as a uniform sprinkle: territory, migration, and contact emerge from geography.
        tribes = max(1, self.cfg.house_tribes)
        per = [n // tribes] * tribes
        for i in range(n % tribes):
            per[i] += 1

        def torus_dist(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            dx = min(dx, self.cfg.width - dx)
            dy = min(dy, self.cfg.height - dy)
            return math.hypot(dx, dy)

        # settlements land anywhere — purely random anchors, kept apart by a minimum
        # toroidal separation so villages never spawn on top of each other
        min_sep = 0.7 * math.sqrt(self.cfg.width * self.cfg.height / tribes)
        anchors: list[tuple[int, int]] = []
        while len(anchors) < tribes:
            best, best_d = None, -1.0
            for _ in range(60):
                cand = (
                    self.rng.randrange(self.cfg.width),
                    self.rng.randrange(self.cfg.height),
                )
                d = min((torus_dist(cand, a) for a in anchors), default=min_sep + 1)
                if d >= min_sep:
                    best = cand
                    break
                if d > best_d:
                    best, best_d = cand, d
            anchors.append(best)
        for t in range(tribes):
            lineage = self.new_lineage()
            self.register_tribe(lineage, f"house:{HOUSE_NAMES[t % len(HOUSE_NAMES)]}")
            # one shared randomized strand per tribe: every founder cell starts with the SAME
            # complete DNA, so a tribe is genetically uniform until mutation and selection diverge it
            g = random_genome(self.rng, self.cfg.max_genes)
            cx, cy = anchors[t]
            rad = max(3, int(math.sqrt(per[t]) * 1.6))
            placed, attempts = 0, 0
            while placed < per[t] and attempts < per[t] * 80:
                attempts += 1
                q = (cx + self.rng.randint(-rad, rad)) % self.cfg.width
                r = (cy + self.rng.randint(-rad, rad)) % self.cfg.height
                cell = self.cells.get((q, r))
                if cell is not None and cell.organism is None:
                    self.spawn(q, r, g, lineage, self.cfg.birth_energy)
                    placed += 1

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
