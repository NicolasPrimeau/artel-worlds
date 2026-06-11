from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .arena import Arena
from .config import DEFAULT
from .control import decide

STATIC = Path(__file__).parent / "static"
TICK_INTERVAL = 1.0


class Phalanx:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.tokens: dict[str, str] = {}  # team -> secret
        self._new_match()

    def _new_match(self) -> None:
        seed = self.arena.rng.randint(1, 1_000_000) if hasattr(self, "arena") else 1
        self.arena = Arena(DEFAULT, seed=seed)
        self.arena.seed_house()
        self.tokens.clear()

    def has_players(self) -> bool:
        return any(k.startswith("player:") for k in self.arena.team_kind.values())

    def is_player_team(self, team: str) -> bool:
        return self.arena.team_kind.get(team, "").startswith("player:")

    def snapshot(self) -> dict:
        a = self.arena
        return {
            **a.stats(),
            "width": a.cfg.width,
            "height": a.cfg.height,
            "tank_list": [
                {
                    "id": t.id,
                    "x": t.x,
                    "y": t.y,
                    "heading": t.heading,
                    "gun": t.gun,
                    "energy": round(t.energy),
                    "team": t.team,
                    "player": self.is_player_team(t.team),
                }
                for t in a.tanks.values()
            ],
            "shell_list": [{"x": s.x, "y": s.y, "team": s.team} for s in a.shells],
            "wall_list": [{"x": x, "y": y} for (x, y) in a.walls],
        }


G = Phalanx()


class Join(BaseModel):
    agent_id: str


class Intent(BaseModel):
    turn: int = 0
    move: str = "hold"
    aim: int | None = None
    fire: float = 0.0


class TeamIntent(BaseModel):
    actions: dict[str, Intent]


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
        if bool(G.viewers) or G.has_players():
            async with G.lock:
                # any tank without a submitted order runs its default brain (instinct)
                for t in list(G.arena.tanks.values()):
                    if t.id not in G.arena.pending:
                        p = G.arena.perceive(t.id)
                        if p:
                            G.arena.submit(t.id, decide(p, G.arena.cfg))
                G.arena.step()
                if G.arena.winner is not None or len(G.arena.teams_alive()) <= 1:
                    G._new_match()
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


app = FastAPI(title="Phalanx — Artel Worlds", lifespan=_lifespan)


def _token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-team-token", "")


def _authorize(team: str, request: Request) -> None:
    if not G.is_player_team(team):
        raise HTTPException(404, "no such player team")
    expected = G.tokens.get(team, "")
    if not expected or not secrets.compare_digest(expected, _token(request)):
        raise HTTPException(401, "this team isn't yours — pass your join token as Bearer")


@app.post("/join")
async def join(body: Join):
    async with G.lock:
        team = body.agent_id
        if team in G.arena.team_kind:
            raise HTTPException(409, "that team name is taken")
        anchor = (G.arena.cfg.width // 2, G.arena.cfg.height // 2)
        ids = G.arena.add_team(team, f"player:{body.agent_id}", anchor)
        token = secrets.token_urlsafe(16)
        G.tokens[team] = token
    return {"agent_id": body.agent_id, "team": team, "token": token, "tanks": ids}


@app.get("/team/{team}/perceive")
async def team_perceive(team: str, request: Request):
    async with G.lock:
        _authorize(team, request)
        tanks = G.arena.team_tanks(team)
        if not tanks:
            raise HTTPException(404, "your team has no tanks left — POST /join for the next match")
        return {
            "team": team,
            "size": len(tanks),
            "tanks": {str(t.id): G.arena.perceive(t.id) for t in tanks},
        }


@app.post("/team/{team}/intend")
async def team_intend(team: str, body: TeamIntent, request: Request):
    async with G.lock:
        _authorize(team, request)
        applied = 0
        for tid_s, intent in body.actions.items():
            try:
                tid = int(tid_s)
            except ValueError:
                continue
            t = G.arena.tanks.get(tid)
            if t is not None and t.team == team:
                G.arena.submit(tid, intent.model_dump())
                applied += 1
    return {"applied": applied}


@app.get("/health")
async def health():
    return {"status": "ok", "tick": G.arena.tick_count}


@app.get("/state")
async def state():
    async with G.lock:
        return JSONResponse(G.snapshot())


@app.post("/reset")
async def reset():
    async with G.lock:
        G._new_match()
    return {"ok": True}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    G.viewers.add(ws)
    try:
        async with G.lock:
            await ws.send_text(json.dumps(G.snapshot()))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        G.viewers.discard(ws)


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "Phalanx", "ui": "static/index.html not built yet"}
