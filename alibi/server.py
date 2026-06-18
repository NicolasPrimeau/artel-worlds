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

from . import artel, llm
from .engine import MAX_TICKS, new_game
from .meeting import (
    CREW_POOL,
    THING_MODEL,
    assign_models,
    run_canned_meeting,
    run_llm_meeting,
)

# Alibi runs one game after another, but ONLY while someone is watching (free-tier Groq, like phalanx):
# no viewers → no ticks, no LLM calls. A game is a task phase (agents wander the station, the Thing
# kills) punctuated by meetings — which are streamed statement-by-statement so the chat builds live on
# the page. Crew win by clearing the task board or ejecting the Thing; the Thing wins at parity.

_rng = SystemRandom()
log = logging.getLogger("alibi")

STATIC = Path(__file__).parent / "static"
TASK_TICK = float(os.environ.get("ALIBI_TICK_INTERVAL", "3.0"))  # min seconds per task-phase tick
STMT_DELAY = float(os.environ.get("ALIBI_STMT_DELAY", "3.4"))  # seconds each spoken line holds
PRE_VOTE = float(os.environ.get("ALIBI_PRE_VOTE", "3.5"))  # the table settles before the vote opens
VOTE_DELAY = float(os.environ.get("ALIBI_VOTE_DELAY", "1.9"))  # seconds between revealed votes
EJECT_WALK = (
    4.0  # the ejected researcher is walked into the airlock — BEFORE we reveal what they were
)
EJECT_REVEAL = 4.0  # then hold on the human/Thing reveal
GAMEOVER_LINGER = 8.0  # hold on the final board before the next game
_ADMIN_TOKEN = os.environ.get("WORLDS_ADMIN_TOKEN", "")
N_AGENTS = int(os.environ.get("ALIBI_AGENTS", "10"))
N_IMPOSTORS = int(os.environ.get("ALIBI_IMPOSTORS", "2"))


def _state_path() -> Path:
    p = Path(os.environ.get("ALIBI_STATE", "/data/alibi_state.json"))
    return p if p.parent.exists() else Path(p.name)


def _task_rooms(g) -> dict:
    # rooms with a lit console: unclaimed tasks on the board PLUS tasks in progress (a crew walking to
    # one, or working it) — so the map shows the whole task load, not just the spare consoles.
    rooms: dict = {}
    for r in g.open_tasks:
        rooms[r] = rooms.get(r, 0) + 1
    for a in g.living(impostor=False):
        if a.work > 0:
            rooms[a.room] = rooms.get(a.room, 0) + 1
        elif a.dest is not None:
            rooms[a.dest] = rooms.get(a.dest, 0) + 1
    return rooms


class Alibi:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.scores = {"crew": 0, "thing": 0}  # games won by each side
        self.recent: list[dict] = []  # [{win: crew-won?}] for the ops dots
        self.completed = 0
        self.paused = False
        self.phase = "task"  # task | meeting | vote | ejection | gameover
        self.revealed = False  # during ejection: has the human/Thing reveal happened yet?
        self.meeting = None
        self.task_q: asyncio.Queue = asyncio.Queue()  # engine task events → mirrored onto Artel
        self.task_pool: dict = {}  # room -> [open Artel task ids waiting to be claimed]
        self.task_working: dict = {}  # agent id -> the Artel task id it's currently doing
        self._restore_state()
        self._new_game()

    def set_paused(self, value: bool) -> None:
        self.paused = bool(value)
        self.persist_state()

    def enqueue_task_events(self) -> None:
        for ev in self.g.events:
            self.task_q.put_nowait(ev)

    async def reset_artel(self) -> None:
        # restart the project on Artel for the new game: wipe its tasks/messages, drop stale state,
        # then queue the freshly-seeded board so it's re-created as Artel tasks.
        await artel.clear_project()
        self.task_pool.clear()
        self.task_working.clear()
        while not self.task_q.empty():
            try:
                self.task_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.enqueue_task_events()

    def _new_game(self) -> None:
        self.g = new_game(_rng.randint(1, 2**31 - 1), n=N_AGENTS, impostors=N_IMPOSTORS)
        assign_models(self.g)
        self.tasks_total = self.g.tasks_goal
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
        # the Thing is unmasked only at the dramatic reveal (after the walk-out) or on the final board
        reveal = (self.phase == "ejection" and self.revealed) or self.phase == "gameover"
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
            if self.phase in (
                "vote",
                "ejection",
            ):  # reveal the ballot as it fills, one vote at a time
                meeting["votes"] = {
                    g.by_id(v).name: ("skip" if t == -1 else g.by_id(t).name)
                    for v, t in (mt.votes or {}).items()
                }
            if self.phase == "ejection" and mt.ejected is not None:
                meeting["ejected"] = g.by_id(
                    mt.ejected
                ).name  # who walks out (shown during the walk)
                if self.revealed:  # ...but WHAT they were is withheld until the reveal beat
                    meeting["ejected_was_thing"] = g.by_id(mt.ejected).impostor
        return {
            "phase": self.phase,
            "revealed": self.revealed,
            "round": len(g.meetings) + 1,
            "tick": g.tick,
            "tasksDone": g.tasks_done,
            "tasksTotal": g.tasks_goal,
            "alive": len(g.living()),
            "total": len(g.agents),
            "agents": agents,
            "station": {
                "outpost": g.outpost,
                "rects": {n: list(r) for n, r in g.rects.items()},
                "corridor": [[t[0], t[1]] for t in g.corridor],
                "centers": {n: [round(c[0], 2), round(c[1], 2)] for n, c in g.centers.items()},
                "doors": [[a, b, dx, dy] for (a, b), (dx, dy) in g.doors.items()],
                "vents": sorted({tuple(sorted((a, b))) for a, ns in g.vents.items() for b in ns}),
                "openTasks": _task_rooms(g),  # room -> lit consoles (unclaimed + in-progress)
            },
            "meeting": meeting,
            "winner": g.winner,
            "win_by": g.win_by,
            "scores": dict(self.scores),
            "paused": self.paused,
            "caption": self.caption(),
            "lastKill": (
                {
                    "tick": g.last_kill["tick"],
                    "victim": g.by_id(g.last_kill["victim"]).name,
                    "room": g.last_kill["room"],
                }
                if g.last_kill
                else None
            ),
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
    G.g.reconvene()  # everyone — task-workers included — downs tools and gathers at the Mess Hall table
    G.phase = "meeting"
    G.revealed = False
    G.meeting = mt
    await _broadcast()

    async def on_item(kind: str, agent_id: int, payload) -> None:
        # paces the meeting: deliberation lines hold on screen, the table SETTLES before the vote opens,
        # then votes land one at a time. Each line/vote is mirrored onto the Artel bus.
        if kind == "settle":  # deliberation over → a beat before the vote
            G.phase = "vote"
            await _broadcast()
            await asyncio.sleep(PRE_VOTE)
            return
        name = G.g.by_id(agent_id).name
        if kind == "statement":
            G.phase = "meeting"
            await artel.say(agent_id, name, payload)
            await _broadcast()
            await asyncio.sleep(STMT_DELAY)
        else:
            G.phase = "vote"
            target = G.g.by_id(payload).name if payload is not None and payload >= 0 else "abstains"
            await artel.say(agent_id, name, f"votes {target}.", subject="alibi-vote")
            await _broadcast()
            await asyncio.sleep(VOTE_DELAY)

    if llm.enabled():
        votes = await run_llm_meeting(G.g, mt, on_item)
    else:  # no key configured → canned statements + deterministic decider so the scene still plays
        from .brain import make_decider

        votes = await run_canned_meeting(G.g, mt, make_decider(share=True), on_item)
    G.g.apply_votes(mt, votes)
    # ejection: walk the ejected out (suspense), THEN reveal whether they were the Thing
    G.phase = "ejection"
    G.revealed = False
    await _broadcast()
    if mt.ejected is not None:
        await asyncio.sleep(EJECT_WALK)
    G.revealed = True  # the airlock reveal
    await _broadcast()
    await asyncio.sleep(EJECT_REVEAL)


_TASK_VERBS = (
    "Run diagnostics in",
    "Reroute power in",
    "Recalibrate",
    "Service the rig in",
    "Log readings in",
)


async def _mirror_event(ev) -> None:
    # turn one engine task event into a real Artel task action (create on spawn, claim by the agent,
    # complete when done). Runs in a single serial worker, so task_pool/task_working never race.
    kind = ev[0]
    if kind == "spawn":
        room = ev[1]
        tid = await artel.create_task(f"{_TASK_VERBS[hash(room) % len(_TASK_VERBS)]} the {room}")
        if tid:
            G.task_pool.setdefault(room, []).append(tid)
    elif kind == "claim":
        _, agent_id, room = ev
        pool = G.task_pool.get(room) or []
        tid = pool.pop(0) if pool else await artel.create_task(f"Work the {room}")
        if tid:
            await artel.claim_task(agent_id, tid)
            G.task_working[agent_id] = tid
    elif kind == "complete":
        tid = G.task_working.pop(ev[1], None)
        if tid:
            await artel.complete_task(ev[1], tid)


async def _task_worker():
    while True:
        ev = await G.task_q.get()
        try:
            await _mirror_event(ev)
        except Exception as e:
            log.warning("task mirror failed: %s", e)


async def _game_loop():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        if not G.paused and G.viewers:
            async with G.lock:
                mt = G.g.step()
                G.enqueue_task_events()  # mirror this tick's task spawns/claims/completes onto Artel
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
                    await G.reset_artel()  # restart the project on Artel for the next game
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
    worker = asyncio.create_task(_task_worker())
    await G.reset_artel()  # clear the project + queue the opening board for the first game
    task = asyncio.create_task(_game_loop())
    try:
        yield
    finally:
        task.cancel()
        worker.cancel()
        for t in (task, worker):
            with contextlib.suppress(asyncio.CancelledError):
                await t
        G.persist_state()
        await artel.aclose()


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
        "artel": artel.status(),
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC / "favicon.ico")


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "Alibi", "ui": "static/index.html not built yet"}
