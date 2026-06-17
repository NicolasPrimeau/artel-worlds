from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
from random import SystemRandom

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import llm
from .engine import MAX_TICKS, ROOMS, new_game
from .meeting import CREW_POOL, THING_MODEL, assign_models, run_llm_meeting

# Alibi runs one game after another, but ONLY while someone is watching (free-tier Groq, like phalanx):
# no viewers → no ticks, no LLM calls. A game is a task phase (agents wander the station, the Thing
# kills) punctuated by meetings — which are streamed statement-by-statement so the chat builds live on
# the page. Crew win by clearing the task board or ejecting the Thing; the Thing wins at parity.

_rng = SystemRandom()
log = logging.getLogger("alibi")

STATIC = Path(__file__).parent / "static"
TASK_TICK = float(os.environ.get("ALIBI_TICK_INTERVAL", "1.4"))  # min seconds per task-phase tick
EJECT_LINGER = 6.0  # hold on the vote + airlock reveal so viewers read the result
GAMEOVER_LINGER = 8.0  # hold on the final board before the next game
_ADMIN_TOKEN = os.environ.get("WORLDS_ADMIN_TOKEN", "")
N_AGENTS = int(os.environ.get("ALIBI_AGENTS", "6"))
N_IMPOSTORS = int(os.environ.get("ALIBI_IMPOSTORS", "1"))


def _state_path() -> Path:
    p = Path(os.environ.get("ALIBI_STATE", "/data/alibi_state.json"))
    return p if p.parent.exists() else Path(p.name)


class Alibi:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.scores = {"crew": 0, "thing": 0}  # games won by each side
        self.recent: list[dict] = []  # [{win: crew-won?}] for the ops dots
        self.completed = 0
        self.paused = False
        self.phase = "task"  # task | meeting | vote | ejection | gameover
        self.meeting = None
        self._restore_state()
        self._new_game()

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)
        self.persist_state()

    def _new_game(self) -> None:
        self.g = new_game(_rng.randint(1, 2**31 - 1), n=N_AGENTS, impostors=N_IMPOSTORS)
        assign_models(self.g)
        self.tasks_total = sum(a.tasks for a in self.g.agents if not a.impostor)
        self.phase = "task"
        self.meeting = None

    def _restore_state(self) -> None:
        try:
            raw = json.loads(_state_path().read_text())
        except Exception:
            return
        self.scores = {k: int(raw.get("scores", {}).get(k, 0)) for k in ("crew", "thing")}
        self.recent = list(raw.get("recent") or [])[-12:]
        self.completed = int(raw.get("completed", 0))
        self.paused = bool(raw.get("paused", False))
        sp = raw.get("spend") or {}
        if sp:
            llm.SPEND.update({k: sp.get(k, llm.SPEND[k]) for k in llm.SPEND})

    def persist_state(self) -> None:
        try:
            path = _state_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "scores": self.scores,
                        "recent": self.recent[-12:],
                        "completed": self.completed,
                        "paused": self.paused,
                        "spend": llm.SPEND,
                    }
                )
            )
            tmp.replace(path)
        except Exception as e:
            log.warning("alibi state persist failed: %s", e)

    def record_result(self) -> None:
        g = self.g
        crew_won = g.winner == "crew"
        self.scores["crew" if crew_won else "thing"] += 1
        self.recent.append({"win": crew_won})
        del self.recent[:-12]
        self.completed += 1
        self.persist_state()

    def caption(self) -> str:
        g = self.g
        if self.phase == "gameover":
            return f"{g.winner} won by {g.win_by} · {len(g.living())} left standing"
        if self.meeting is not None and self.phase in ("meeting", "vote", "ejection"):
            if self.meeting.victim is not None:
                return f"meeting: {g.by_id(self.meeting.victim).name} found dead in {self.meeting.room}"
            return f"emergency meeting in {self.meeting.room}"
        return f"task phase · {len(g.living())}/{len(g.agents)} alive · tick {g.tick}"

    def snapshot(self) -> dict:
        g = self.g
        reveal = self.phase in ("ejection", "gameover")  # only unmask the Thing on a reveal
        agents = []
        for a in g.agents:
            d = {
                "id": a.id,
                "name": a.name,
                "model": a.model,
                "room": a.room,
                "alive": a.alive,
                "tasking": a.tasking,
                "body": not a.alive,
                "color": a.id,
            }
            if reveal:
                d["thing"] = a.impostor
            agents.append(d)
        meeting = None
        mt = self.meeting
        if mt is not None and self.phase in ("meeting", "vote", "ejection"):
            meeting = {
                "room": mt.room,
                "reporter": g.by_id(mt.reporter).name if mt.reporter >= 0 else None,
                "victim": g.by_id(mt.victim).name if mt.victim is not None else None,
                "transcript": [
                    {"name": g.by_id(s).name, "model": g.by_id(s).model, "text": t}
                    for s, t in mt.transcript
                ],
                "votes": {},
                "ejected": None,
                "ejected_was_thing": None,
            }
            if self.phase == "ejection":
                meeting["votes"] = {
                    g.by_id(v).name: ("skip" if t == -1 else g.by_id(t).name)
                    for v, t in (mt.votes or {}).items()
                }
                if mt.ejected is not None:
                    meeting["ejected"] = g.by_id(mt.ejected).name
                    meeting["ejected_was_thing"] = g.by_id(mt.ejected).impostor
        return {
            "phase": self.phase,
            "round": len(g.meetings) + 1,
            "tick": g.tick,
            "tasksDone": self.tasks_total - g.tasks_left(),
            "tasksTotal": self.tasks_total,
            "alive": len(g.living()),
            "total": len(g.agents),
            "agents": agents,
            "rooms": list(ROOMS),
            "meeting": meeting,
            "winner": g.winner,
            "win_by": g.win_by,
            "scores": dict(self.scores),
            "paused": self.paused,
            "caption": self.caption(),
        }


G = Alibi()


async def _broadcast(snap: dict | None = None):
    msg = json.dumps(snap if snap is not None else G.snapshot())
    dead = []
    for ws in list(G.viewers):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        G.viewers.discard(ws)


async def _run_meeting(mt) -> None:
    G.phase = "meeting"
    G.meeting = mt
    await _broadcast()

    async def on_update(kind):
        G.phase = "vote" if kind == "vote" else "meeting"
        await _broadcast()

    if llm.enabled():
        votes = await run_llm_meeting(G.g, mt, on_update)
    else:  # no key configured → fall back to the deterministic decider so the world still runs
        from .brain import make_decider

        votes = make_decider(share=True)(G.g, mt)
        mt.votes = votes
    G.g.apply_votes(mt, votes)
    G.phase = "ejection"
    await _broadcast()
    await asyncio.sleep(EJECT_LINGER)


async def _game_loop():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        if not G.paused and G.viewers:
            async with G.lock:
                mt = G.g.step()
                if mt is not None and G.g.winner is None:
                    await _run_meeting(mt)
                if G.g.winner is not None:
                    G.phase = "gameover"
                    G.record_result()
                    await _broadcast()
            if G.phase == "gameover":
                await asyncio.sleep(GAMEOVER_LINGER)
                async with G.lock:
                    G._new_game()
                await _broadcast()
                continue
            if G.g.tick >= MAX_TICKS:  # safety: stalemate → Thing wins
                async with G.lock:
                    G.g.winner, G.g.win_by = "impostor", "timeout"
                continue
            await _broadcast()
        elif G.paused and G.viewers:
            await _broadcast()
        await asyncio.sleep(max(0.0, TASK_TICK - (loop.time() - start)))


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    task = asyncio.create_task(_game_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        G.persist_state()


app = FastAPI(title="Alibi — Artel Worlds", lifespan=_lifespan)


@app.get("/debug")
async def debug():
    # the standard ops block: cost, results, viewers — plus a live caption.
    return {
        "paused": G.paused,
        "viewers": len(G.viewers),
        "live": bool(G.viewers) and llm.enabled(),
        "model": f"{THING_MODEL} (Thing) + {len(CREW_POOL)}-model crew mesh",
        "spend": round(llm.SPEND["usd"], 5),
        "cap": None,
        "spend_days": dict(llm.SPEND["days"]),
        "calls": llm.SPEND["calls"],
        "results": dict(G.scores),
        "recent": G.recent[-10:],
        "caption": G.caption(),
        "completed": G.completed,
        "phase": G.phase,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "tick": G.g.tick, "phase": G.phase}


@app.get("/state")
async def state():
    async with G.lock:
        return JSONResponse(G.snapshot())


@app.post("/admin/pause")
async def admin_pause(request: Request):
    if not _ADMIN_TOKEN or request.headers.get("x-admin-token") != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    body = await request.json()
    async with G.lock:
        G.set_paused(bool(body.get("paused")))
    return {"paused": G.paused}


@app.post("/reset")
async def reset():
    async with G.lock:
        G.scores = {"crew": 0, "thing": 0}
        G.recent = []
        G.completed = 0
        G._new_game()
        G.persist_state()
    return {"ok": True}


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    G.viewers.add(ws)
    try:
        # send the current frame WITHOUT the lock — the loop holds it across a multi-second LLM
        # meeting, and a new viewer must attach instantly (it gets streamed updates regardless).
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
    return {"world": "Alibi", "ui": "static/index.html not built yet"}
