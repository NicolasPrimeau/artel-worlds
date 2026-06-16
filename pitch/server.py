from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .bot import decide
from .engine import Pitch

log = logging.getLogger("pitch")
STATIC = Path(__file__).parent / "static"
TICK_INTERVAL = float(os.environ.get("PITCH_TICK_INTERVAL", "0.1"))  # sim+broadcast rate
FULLTIME_HOLD = float(os.environ.get("PITCH_FULLTIME_HOLD", "6"))  # pause on the final whistle

# Tongue-in-cheek AI x football club names; two are drawn per match. Player names come from a
# pool so every match reads like a real fixture (the parasocial layer — for free).
CLUBS = [
    "Real Latency",
    "Bayer Neural",
    "Inter Lengtho",
    "Paris Saint-GPT",
    "Gradient Galácticos",
    "AC Median",
    "Boca Tensors",
    "Manchester Neural",
    "Sporting Vector",
    "Ajax Overflow",
]
NAMES = [
    "Okafor",
    "Bianchi",
    "Sorensen",
    "Tanaka",
    "Mbeki",
    "Rossi",
    "Novak",
    "Haaland-9",
    "Pirlo-bot",
    "Adeyemi",
    "Kovač",
    "Silva",
    "Ferreira",
    "Yamamoto",
    "Dubois",
    "Hassan",
    "Larsson",
    "Costa",
    "Petrov",
    "Nakamura",
]


class Game:
    def __init__(self) -> None:
        self.viewers: set[WebSocket] = set()
        self.match_no = 0
        self.history: list[dict] = []  # recent final scores, for the ticker
        self._fulltime_at: float | None = None
        self._rng_i = 0
        self.pitch = Pitch()
        self._new_match()

    def _new_match(self) -> None:
        self.match_no += 1
        i = self._rng_i
        self._rng_i += 1
        home = CLUBS[(2 * i) % len(CLUBS)]
        away = CLUBS[(2 * i + 1) % len(CLUBS)]
        roles = ["GK", "LB", "RB", "CM", "ST"]
        hn = [f"{roles[k]} {NAMES[(i * 5 + k) % len(NAMES)]}" for k in range(5)]
        an = [f"{roles[k]} {NAMES[(i * 5 + k + 7) % len(NAMES)]}" for k in range(5)]
        self.pitch = Pitch(seed=1000 + self.match_no)
        self.pitch.setup(hn, an)
        self.home_club, self.away_club = home, away
        self._fulltime_at = None

    def snapshot(self) -> dict:
        c = self.pitch.cfg
        full = self.pitch.tick >= c.match_ticks
        return {
            "length": c.length,
            "width": c.width,
            "goal_width": c.goal_width,
            "match_no": self.match_no,
            "tick": self.pitch.tick,
            "match_ticks": c.match_ticks,
            "fulltime": full,
            "home": {"club": self.home_club, "score": self.pitch.score["home"]},
            "away": {"club": self.away_club, "score": self.pitch.score["away"]},
            "ball": {"x": round(self.pitch.ball.x, 2), "y": round(self.pitch.ball.y, 2)},
            "players": [
                {
                    "id": p.id,
                    "team": p.team,
                    "name": p.name,
                    "role": p.role,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                }
                for p in self.pitch.players
            ],
            "events": self.pitch.events[-6:],
            "history": self.history[-6:],
        }


G = Game()


async def _broadcast(snap: dict) -> None:
    if not G.viewers:
        return
    msg = json.dumps(snap)
    for ws in list(G.viewers):
        try:
            await ws.send_text(msg)
        except Exception:
            G.viewers.discard(ws)


async def _tick_loop() -> None:
    while True:
        start = asyncio.get_event_loop().time()
        if G.viewers:
            if G.pitch.tick < G.pitch.cfg.match_ticks:
                G.pitch.step(decide)
                await _broadcast(G.snapshot())
            else:
                # full time: hold the final frame so viewers see the result, then kick off anew
                if G._fulltime_at is None:
                    G._fulltime_at = start
                    s = G.pitch.score
                    G.history.append(
                        {"home": G.home_club, "away": G.away_club, "h": s["home"], "a": s["away"]}
                    )
                    del G.history[:-6]
                    await _broadcast(G.snapshot())
                elif start - G._fulltime_at >= FULLTIME_HOLD:
                    G._new_match()
                    await _broadcast(G.snapshot())
        await asyncio.sleep(max(0.0, TICK_INTERVAL - (asyncio.get_event_loop().time() - start)))


async def _lifespan(app: FastAPI):
    t = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        t.cancel()


app = FastAPI(title="Pitch — Artel Worlds", lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "tick": G.pitch.tick, "match": G.match_no}


@app.get("/debug")
async def debug():
    return {"viewers": len(G.viewers), "match": G.match_no, **G.snapshot()}


@app.get("/state")
async def state():
    return JSONResponse(G.snapshot())


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
    except Exception:
        pass
    finally:
        G.viewers.discard(ws)


@app.get("/")
async def root():
    return FileResponse(STATIC / "index.html")
