from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .world import World

log = logging.getLogger("watchtower")
STATIC = Path(__file__).parent / "static"

G = World()


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    loop = asyncio.create_task(G.loop())
    try:
        yield
    finally:
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop
        await G.aclose()


app = FastAPI(title="Watchtower — Artel Worlds", lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "cursor": G.cursor}


@app.get("/debug")
async def debug():
    return G.status()


@app.get("/state")
async def state():
    return JSONResponse(G.snapshot())


@app.get("/metrics")
async def metrics():
    return {
        "summary": G.metrics.summary(),
        "wedge": G.metrics.wedge(),
        "per_family": G.metrics.per_family(),
        "recent": G.metrics.recent(40),
    }


@app.post("/reset")
async def reset():
    # wipe the MTTR curve and restart the paired A/B from zero (operator action)
    await G.reset()
    return {"ok": True, "cursor": G.cursor}


@app.post("/fire")
async def fire():
    # force the next incident immediately — for demos and the smoke test; the loop fires on its own
    if not G.enabled:
        return {"fired": False, "reason": "world disabled (no llm or no artel agents)"}
    asyncio.create_task(G.fire())
    return {"fired": True, "seq": G.cursor}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    G.viewers.add(ws)
    try:
        await ws.send_text(json.dumps(G.snapshot()))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        G.viewers.discard(ws)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    p = STATIC / "favicon.ico"
    return FileResponse(p) if p.exists() else JSONResponse({}, status_code=404)


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "Watchtower", "ui": "static/index.html not built yet"}
