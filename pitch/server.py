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
from .tournament import Tournament

log = logging.getLogger("pitch")
STATIC = Path(__file__).parent / "static"
TICK_INTERVAL = float(os.environ.get("PITCH_TICK_INTERVAL", "0.08"))  # sim+broadcast rate
FULLTIME_HOLD = float(os.environ.get("PITCH_FULLTIME_HOLD", "6"))  # pause on the final whistle


def _rate(mult: float) -> int:
    # map an attribute multiplier (~0.85..1.15) to a FIFA-style 2-digit rating for display
    return max(48, min(99, round(74 + (mult - 1.0) * 160)))


class Game:
    def __init__(self) -> None:
        self.viewers: set[WebSocket] = set()
        self.match_no = 0
        self.history: list[dict] = []  # recent final scores, for the ticker
        self._fulltime_at: float | None = None
        self._champion_done = False
        self.edition = 0
        self.pitch = Pitch()
        self._new_edition()
        self._new_match()

    def _new_edition(self) -> None:
        self.edition += 1
        self.tour = Tournament(edition=self.edition, seed=1000 + self.edition)

    def _new_match(self) -> None:
        tie = self.tour.current()
        if tie is None:  # edition finished — draw the next World Cup
            self._new_edition()
            tie = self.tour.current()
        self.match_no += 1
        self.home_club, self.away_club = tie.a, tie.b
        self.round_label = tie.rnd
        self.pitch = Pitch(seed=2000 + self.match_no)
        self.pitch.setup(self.tour.roster_names(tie.a), self.tour.roster_names(tie.b))
        self._fulltime_at = None
        self._champion_done = False
        self._last_h = self._last_a = 0

    def note_goals(self) -> None:
        # attribute each new goal to its scorer's club for the Golden Boot
        h, a = self.pitch.score["home"], self.pitch.score["away"]
        if (h > self._last_h or a > self._last_a) and self.pitch.scorer:
            club = self.home_club if self.pitch.goal_team == "home" else self.away_club
            if club:
                self.tour.record_goal(club, self.pitch.scorer)
        self._last_h, self._last_a = h, a

    def record_fulltime(self) -> None:
        self.tour.record_result(self.pitch.score["home"], self.pitch.score["away"])
        self._champion_done = self.tour.current() is None

    def tour_snapshot(self) -> dict:
        t = self.tour
        cur = t.order[t.cur] if t.cur < len(t.order) else None
        rounds = [
            [
                {
                    "rnd": tie.rnd,
                    "slot": tie.slot,
                    "a": tie.a,
                    "b": tie.b,
                    "sa": tie.sa,
                    "sb": tie.sb,
                    "pa": tie.pa,
                    "pb": tie.pb,
                    "winner": tie.winner,
                    "played": tie.played,
                    "live": (ri, si) == cur,
                }
                for si, tie in enumerate(rnd)
            ]
            for ri, rnd in enumerate(t.rounds)
        ]
        scorers = sorted(t.scorers.values(), key=lambda r: -r["goals"])[:12]
        return {
            "edition": t.edition,
            "champion": t.champion,
            "rounds": rounds,
            "scorers": scorers,
            "teams": t.standings(),
        }

    def snapshot(self) -> dict:
        c = self.pitch.cfg
        full = self.pitch.tick >= c.match_ticks
        return {
            "length": c.length,
            "width": c.width,
            "goal_width": c.goal_width,
            "match_no": self.match_no,
            "edition": self.edition,
            "round": self.round_label,
            "tick": self.pitch.tick,
            "match_ticks": c.match_ticks,
            "fulltime": full,
            "champion": self.tour.champion,
            "celebrating": self.pitch.celebrate > 0,
            "scorer": self.pitch.scorer,
            "goal_team": self.pitch.goal_team,
            "restart": self.pitch.restart_kind,
            "home": {
                "club": self.home_club,
                "score": self.pitch.score["home"],
                "formation": "-".join(map(str, self.pitch.shapes.get("home", ()))),
            },
            "away": {
                "club": self.away_club,
                "score": self.pitch.score["away"],
                "formation": "-".join(map(str, self.pitch.shapes.get("away", ()))),
            },
            "ball": {"x": round(self.pitch.ball.x, 2), "y": round(self.pitch.ball.y, 2)},
            "players": [
                {
                    "id": p.id,
                    "team": p.team,
                    "name": p.name,
                    "role": p.role,
                    "num": p.number,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "pac": _rate(p.pace),
                    "pas": _rate(p.acc),
                    "sho": _rate(p.finishing),
                    "ctl": _rate(p.control),
                    "str": _rate(p.strength),
                    "han": _rate(p.handling) if p.role == "GK" else None,
                }
                for p in self.pitch.players
            ],
            "events": self.pitch.events[-6:],
            "history": self.history[-6:],
            "tournament": self.tour_snapshot(),
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
                G.note_goals()
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
                    G.record_fulltime()  # advance the bracket so it updates during the hold
                    await _broadcast(G.snapshot())
                else:
                    # linger longer on a final so the champion is savoured before the next draw
                    hold = FULLTIME_HOLD * 2 if G._champion_done else FULLTIME_HOLD
                    if start - G._fulltime_at >= hold:
                        if G._champion_done:
                            G._new_edition()
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
    # no-cache so a phone always revalidates and never runs a stale build of the inline client
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
