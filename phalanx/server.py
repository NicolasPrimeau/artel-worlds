from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
from pathlib import Path
from random import SystemRandom

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .agent import Squad
from .arena import Arena
from .config import DEFAULT
from .control import Bot

_rng = SystemRandom()

STATIC = Path(__file__).parent / "static"
TICK_INTERVAL = 1.0  # one tick per second — easy to follow
# The whole demo: same arena, same guns. Artel is three real LLM agents — one per tank —
# coordinating live over artel.run; Red is deterministic seek-and-destroy bots that can't
# talk to each other. The only thing Artel has that Red doesn't is each other, through
# Artel. Nothing about Artel's play is scripted: the models drive every move.
COORDINATED = {"artel"}


class Phalanx:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.tokens: dict[str, str] = {}  # team -> secret
        self.scores: dict[str, int] = {}  # wins, accumulated across matches
        self.squad = Squad()  # the live Artel LLM agents (no-op until keys + a viewer)
        self.intents: dict[int, dict] = {}  # tank id -> latest LLM decision, applied each tick
        self._squad_match = -1  # which match the squad is currently driving
        self._new_match()

    def _new_match(self) -> None:
        self.match_no = getattr(self, "match_no", -1) + 1
        seed = _rng.randint(1, 2**31 - 1)  # fresh random map layout each match
        self.arena = Arena(DEFAULT, seed=seed)
        self.arena.seed_house(flip=bool(self.match_no % 2))  # alternate corners, fair series
        self.tokens.clear()
        self.intents = {}
        self.squad.stop()
        self._squad_match = -1
        # one Bot per tank. Coordinated teams (Artel) share a single board across all
        # their tanks — that shared knowledge IS the coordination edge; every other team
        # gets a private board per tank, so it only ever knows what it personally saw.
        boards: dict[str, dict] = {}
        self.bots = {}
        for t in self.arena.tanks.values():
            coord = t.team in COORDINATED
            board = boards.setdefault(t.team, {}) if coord else {}
            self.bots[t.id] = Bot(t.id, t.team, board, coord)

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
            "scores": self.scores,
            "coordinated": list(COORDINATED),
            "zone": {
                "q": a.cfg.width // 2,
                "r": a.cfg.height // 2,
                "radius": round(a.safe_radius(), 2),
            },
            "tank_list": [
                {
                    "id": t.id,
                    "q": t.q,
                    "r": t.r,
                    "heading": t.heading,
                    "energy": round(t.energy),
                    "team": t.team,
                    "player": self.is_player_team(t.team),
                    "coord": t.team in COORDINATED,
                }
                for t in a.tanks.values()
            ],
            "tracer_list": [
                {"q": s["q"], "r": s["r"], "tq": s["tq"], "tr": s["tr"], "team": s["team"]}
                for s in a.tracers
            ],
            "wall_list": [{"q": q, "r": r} for (q, r) in a.walls],
        }


G = Phalanx()


class Join(BaseModel):
    agent_id: str


class Intent(BaseModel):
    turn: int = 0
    move: str = "hold"
    fire: int = 0  # enemy tank id to shoot at (0 = hold fire)


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


def _artel_alive_ids() -> list[int]:
    return sorted(t.id for t in G.arena.tanks.values() if t.team in COORDINATED)


def _manage_squad() -> None:
    # the live LLM agents only run while someone's watching and the spend cap allows —
    # no one watching, no spend. Each new match restarts them on that match's Artel tanks.
    if not G.viewers or not G.squad.enabled:
        if G._squad_match != -1:
            G.squad.stop()
            G._squad_match = -1
        return
    if G._squad_match != G.match_no:
        G.squad.start(
            _artel_alive_ids(),
            lambda tid: G.arena.perceive(tid),
            lambda tid, intent: G.intents.__setitem__(tid, intent),
        )
        G._squad_match = G.match_no


async def _tick_loop():
    while True:
        if bool(G.viewers) or G.has_players():
            async with G.lock:
                a = G.arena
                _manage_squad()
                live_artel = G.squad.enabled and G._squad_match == G.match_no
                for t in list(a.tanks.values()):
                    if t.id in a.pending or G.is_player_team(t.team):
                        continue
                    if t.team in COORDINATED and live_artel:
                        a.submit(t.id, G.intents.get(t.id, {}))  # carry the LLM's last call
                        continue
                    p = a.perceive(t.id)
                    if not p:
                        continue
                    bot = G.bots.get(t.id)
                    if bot is None:
                        continue
                    a.submit(t.id, bot.decide(p, a.cfg, a.tick_count))
                a.step()
                if a.winner is not None or len(a.teams_alive()) <= 1:
                    if a.winner:
                        G.scores[a.winner] = G.scores.get(a.winner, 0) + 1
                    G._new_match()
                snap = G.snapshot()
            await _broadcast(snap)
        await asyncio.sleep(TICK_INTERVAL)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    tick = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        G.squad.stop()
        tick.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick


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
