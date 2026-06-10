from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .agent import HeuristicAgent
from .config import DEFAULT
from .genome import random_genome
from .tick import step
from .world import World

STATIC = Path(__file__).parent.parent / "static"

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


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "artel-worlds #1", "ui": "static/index.html not built yet"}
