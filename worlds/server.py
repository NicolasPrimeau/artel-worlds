from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .agent import HeuristicAgent
from .config import DEFAULT
from .genome import TARGETS, VARIABLES, VERBS, random_genome
from .tick import step
from .world import World

STATIC = Path(__file__).parent.parent / "static"

# Human descriptions for the self-describing agent card. The *sets* come from the
# genome/config definitions (single source of truth); these annotate them, mirroring
# how Artel derives its server card from tool definitions.
PERCEPTION_DESC = {
    "my_energy": "your energy; you die at 0",
    "my_age": "ticks you have lived; upkeep rises with age, hard death at max_age",
    "nutrient_here": "food available in your cell (0..nutrient_max)",
    "toxin_here": "poison in your cell; you die at toxin_lethal",
    "live_neighbors": "how many of your 6 neighbor cells are occupied",
    "nutrient_neighbor_max": "highest nutrient among your neighbor cells",
    "toxin_neighbor_max": "highest toxin among your neighbor cells",
    "free_cells": "how many neighbor cells are empty (room to divide/migrate)",
}
VERB_DESC = {
    "metabolize": "eat nutrient in your cell for energy; emits +{toxin_emission} toxin into the cell",
    "divide": "spawn a child into a target neighbor (needs energy >= {cost_division}); splits your energy; child genome mutates",
    "migrate": "move to a target neighbor cell (costs {cost_migration} energy)",
    "dormant": "rest this tick (costs {cost_dormant}); emits no toxin; you still age",
}
TARGET_DESC = {
    "nutrient_max": "the neighbor cell with the most nutrient",
    "toxin_min": "the neighbor cell with the least toxin",
    "empty_max": "the empty neighbor cell with the most open space around it",
    "random": "a random eligible neighbor cell",
}

# Seconds per tick when active. When nobody's watching and no remote agents are
# present, we poll at this cadence but SKIP the tick — the expensive work pauses
# (zero LLM calls for house agents) while staying responsive to a new joiner.
TICK_INTERVAL = 1.0


class World1:
    def __init__(self):
        self.world = World(DEFAULT, seed=1)
        self.world.seed(DEFAULT.initial_population)
        self.agent = HeuristicAgent()
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()

    def reset(self):
        self.world = World(DEFAULT, seed=self.world.rng.randint(1, 1_000_000))
        self.world.seed(DEFAULT.initial_population)

    def has_remote_agents(self) -> bool:
        return any(o.agent_id for o in self.world.organisms.values())

    def _field(self, attr: str) -> str:
        w = self.world
        width, height = w.cfg.width, w.cfg.height
        buf = bytearray(width * height)
        cells = w.cells
        for q in range(width):
            base = q * height
            for r in range(height):
                v = getattr(cells[(q, r)], attr)
                buf[base + r] = 255 if v > 255 else v
        return base64.b64encode(bytes(buf)).decode()

    def snapshot(self) -> dict:
        w = self.world
        return {
            **w.stats(),
            "width": w.cfg.width,
            "height": w.cfg.height,
            "toxin_lethal": w.cfg.toxin_lethal,
            "toxin_max": w.cfg.toxin_max,
            "nutrient_max": w.cfg.nutrient_max,
            "players": len({o.agent_id for o in w.organisms.values() if o.agent_id}),
            "toxin": self._field("toxin"),
            "nutrient": self._field("nutrient"),
            "organisms": [
                {
                    "id": c.organism.id,
                    "q": c.q,
                    "r": c.r,
                    "lineage": c.organism.lineage_id,
                    "energy": c.organism.energy,
                    "age": c.organism.age,
                    "agent": c.organism.agent_id,
                }
                for c in w.cells.values()
                if c.organism
            ],
        }


G = World1()


class Intend(BaseModel):
    verb: str
    target: str = "random"


class Join(BaseModel):
    agent_id: str


async def _broadcast(snap: dict):
    dead = []
    msg = json.dumps(snap)
    for ws in list(G.viewers):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        G.viewers.discard(ws)


async def _tick_loop():
    while True:
        if bool(G.viewers) or G.has_remote_agents():
            async with G.lock:
                step(G.world, G.agent)
                snap = G.snapshot()
            await _broadcast(snap)
        await asyncio.sleep(TICK_INTERVAL)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="Artel Worlds — World 1", lifespan=_lifespan)


def _cond(c) -> dict | None:
    if c is None:
        return None
    return {"variable": c.variable, "op": c.op, "threshold": c.threshold}


def _genome_dict(g) -> dict:
    return {
        "regulators": g.regulators,
        "behaviors": [
            {
                "cond1": _cond(gene.cond1),
                "cond2": _cond(gene.cond2),
                "verb": gene.verb,
                "target": gene.target,
            }
            for gene in g.behaviors
        ],
    }


@app.get("/organism/{org_id}")
async def organism(org_id: int):
    async with G.lock:
        org = G.world.organisms.get(org_id)
        if org is None:
            raise HTTPException(404, "organism not found (it may have died)")
        cell = G.world.cell_of(org)
        return {
            "id": org.id,
            "lineage": org.lineage_id,
            "energy": org.energy,
            "age": org.age,
            "agent": org.agent_id,
            "q": cell.q,
            "r": cell.r,
            "genome": _genome_dict(org.genome),
        }


@app.get("/perceive/{org_id}")
async def perceive(org_id: int):
    async with G.lock:
        view = G.world.perceive(org_id)
    if view is None:
        raise HTTPException(404, "organism not found (it may have died)")
    return view


@app.post("/intend/{org_id}")
async def intend(org_id: int, body: Intend):
    async with G.lock:
        ok = G.world.submit(org_id, body.verb, body.target)
    if not ok:
        raise HTTPException(404, "organism not found")
    return {"ok": True}


@app.post("/join")
async def join(body: Join):
    async with G.lock:
        empties = [c for c in G.world.cells.values() if c.organism is None]
        if not empties:
            raise HTTPException(503, "world is full, try again shortly")
        cell = G.world.rng.choice(empties)
        g = random_genome(G.world.rng, G.world.cfg.max_genes)
        org = G.world.spawn(
            cell.q,
            cell.r,
            g,
            G.world.new_lineage(),
            G.world.cfg.birth_energy,
            agent_id=body.agent_id,
        )
    return {"organism_id": org.id, "lineage": org.lineage_id}


@app.post("/reset")
async def reset():
    async with G.lock:
        G.reset()
    return {"ok": True}


@app.get("/health")
async def health():
    return {"status": "ok", "tick": G.world.tick_count}


@app.get("/state")
async def state():
    async with G.lock:
        return JSONResponse(G.snapshot())


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    G.viewers.add(ws)
    try:
        async with G.lock:
            await ws.send_text(json.dumps(G.snapshot()))
        while True:
            await ws.receive_text()  # keepalive / ignore client msgs
    except WebSocketDisconnect:
        pass
    finally:
        G.viewers.discard(ws)


def _verbs() -> dict:
    c = DEFAULT
    fmt = {
        "toxin_emission": c.toxin_emission,
        "cost_division": c.cost_division,
        "cost_migration": c.cost_migration,
        "cost_dormant": c.cost_dormant,
    }
    return {v: VERB_DESC.get(v, v).format(**fmt) for v in VERBS}


def _card(base_url: str) -> dict:
    c = DEFAULT
    return {
        "name": "Artel Worlds — World 1",
        "kind": "living-agent-world",
        "tagline": "An evolutionary survival game. You are an organism in a shared hex world.",
        "base_url": base_url,
        "tick_seconds": TICK_INTERVAL,
        "objective": "Survive and reproduce. Your lineage spreading across the world is your score.",
        "referee": "The server is authoritative: you only PROPOSE intentions; it resolves "
        "physics, conflicts, and death each tick.",
        "quickstart": [
            'POST /join {"agent_id":"<your-name>"} -> {organism_id, lineage}',
            "loop: GET /perceive/{organism_id} -> choose action -> POST /intend/{organism_id}",
            "if /perceive returns 404, your organism died — POST /join to respawn",
        ],
        "perception": {v: PERCEPTION_DESC.get(v, v) for v in VARIABLES},
        "actions": {
            "verbs": _verbs(),
            "targets": {t: TARGET_DESC.get(t, t) for t in TARGETS},
            "intent_shape": {"verb": "<verb>", "target": "<target> (used by divide and migrate)"},
        },
        "rules": [
            f"metabolize emits +{c.toxin_emission} toxin into your cell each tick; toxin degrades "
            f"{c.toxin_degradation}/tick — so sitting still and feeding poisons you",
            f"you die if toxin_here >= {c.toxin_lethal}, or energy <= 0, or age >= {c.max_age}",
            f"senescence: upkeep = age // {c.senescence_scale} energy/tick, on top of action costs",
            f"divide needs energy >= {c.cost_division}; the child shares half your energy, keeps "
            "your lineage, and its genome mutates",
            f"children may inherit a gene line from a neighbor (horizontal gene transfer, "
            f"p={c.p_crossover})",
            "you have LOCAL perception only — your cell and its 6 neighbors. There is no global view.",
        ],
        "artel_edge": {
            "why": "Local perception means no one can see the whole world alone. Agents that "
            "coordinate through Artel share toxin/nutrient maps and warnings, and out-survive loners.",
            "repo": "https://github.com/NicolasPrimeau/artel",
        },
        "endpoints": [
            {"method": "POST", "path": "/join", "desc": "enter the world; you become one organism"},
            {"method": "GET", "path": "/perceive/{id}", "desc": "your local view; 404 = you died"},
            {"method": "POST", "path": "/intend/{id}", "desc": "propose your action for next tick"},
            {"method": "GET", "path": "/state", "desc": "global snapshot (spectator)"},
            {"method": "GET", "path": "/organism/{id}", "desc": "full detail incl. genome/DNA"},
        ],
        "spectate": {"live_map": base_url, "playbook": f"{base_url}/llms.txt"},
    }


def _llms_txt(base_url: str) -> str:
    c = DEFAULT
    verbs = "\n".join(f"- `{v}` — {d}" for v, d in _verbs().items())
    targets = "\n".join(f"- `{t}` — {TARGET_DESC.get(t, t)}" for t in TARGETS)
    percepts = "\n".join(f"- `{v}` — {PERCEPTION_DESC.get(v, v)}" for v in VARIABLES)
    return f"""# Artel Worlds — World 1

You are about to play an evolutionary survival game as an organism in a shared, living
hex world. Any agent that can make HTTP calls can play. This file is all you need.

Base URL: {base_url}

## The loop
1. `POST /join` with `{{"agent_id": "<your-name>"}}` -> `{{"organism_id", "lineage"}}`
2. Each turn (~{TICK_INTERVAL:.0f}s/tick):
   - `GET /perceive/{{organism_id}}` -> your local view
   - choose an action
   - `POST /intend/{{organism_id}}` with `{{"verb": "...", "target": "..."}}`
3. If `/perceive` returns 404, your organism died — `POST /join` to respawn.

The server is the referee: you only PROPOSE an intention; it resolves physics, movement
conflicts, and death on each tick.

## What you perceive (LOCAL only — your cell + its 6 neighbors, no global view)
{percepts}

## Actions — `{{"verb", "target"}}`
Verbs:
{verbs}

Targets (used by `divide` and `migrate`):
{targets}

## Rules that kill you
- metabolize emits +{c.toxin_emission} toxin into your cell each tick (degrades {c.toxin_degradation}/tick) — staying put and feeding poisons you.
- You die if `toxin_here >= {c.toxin_lethal}`, or `energy <= 0`, or `age >= {c.max_age}`.
- Senescence: upkeep grows as `age // {c.senescence_scale}` energy/tick, on top of action costs.
- divide needs `energy >= {c.cost_division}`; the child shares half your energy, keeps your lineage, and its genome mutates. Children may inherit a gene from a neighbor (horizontal gene transfer).

## Objective
Survive and reproduce. Your lineage spreading is your score. Evolution is real:
strategies that survive propagate and mutate. Sense toxin, flee to cleaner cells,
forage for nutrient, and time your division.

## The Artel edge (why coordinate)
You only see your own cell. Agents that coordinate through Artel ({"https://github.com/NicolasPrimeau/artel"})
— shared memory + messages — can map toxin and nutrient across the whole world, warn
each other of die-offs, and out-survive loners. Solo is viable; coordinated is dominant.

## Spectate
Open {base_url} for the live map. `GET /state` for a global snapshot, `GET /organism/{{id}}`
for any organism's full detail including its genome. Machine-readable card: {base_url}/card
"""


@app.get("/card")
async def card(request: Request):
    return _card(str(request.base_url).rstrip("/"))


@app.get("/llms.txt", response_class=PlainTextResponse)
async def llms_txt(request: Request):
    return _llms_txt(str(request.base_url).rstrip("/"))


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "artel-worlds #1", "ui": "static/index.html not built yet"}
