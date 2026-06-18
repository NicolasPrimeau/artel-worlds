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

from . import artel, autonomy, llm
from .engine import HUB, MAX_TICKS, Meeting, new_game
from .meeting import (
    run_canned_meeting,
    run_llm_meeting,
)

# Alibi runs one game after another, but ONLY while someone is watching (free-tier Groq, like phalanx):
# no viewers → no ticks, no LLM calls. A game is a task phase (agents wander the station, the Cold
# kills) punctuated by meetings — which are streamed statement-by-statement so the chat builds live on
# the page. Crew win by clearing the task board or ejecting the Cold; the Cold wins at parity.

_rng = SystemRandom()
log = logging.getLogger("alibi")

STATIC = Path(__file__).parent / "static"
TASK_TICK = float(os.environ.get("ALIBI_TICK_INTERVAL", "4.2"))  # min seconds per task-phase tick
STMT_DELAY = float(os.environ.get("ALIBI_STMT_DELAY", "4.6"))  # seconds each spoken line holds
PRE_VOTE = float(os.environ.get("ALIBI_PRE_VOTE", "3.5"))  # the table settles before the vote opens
VOTE_DELAY = float(os.environ.get("ALIBI_VOTE_DELAY", "1.9"))  # seconds between revealed votes
WHISPER_DELAY = 1.6  # how long a private-whisper indicator flashes before play moves on
EJECT_WALK = (
    4.0  # the ejected researcher is walked out into the storm — BEFORE we reveal what they were
)
EJECT_REVEAL = 4.0  # then hold on the human/Cold reveal
GAMEOVER_LINGER = 14.0  # hold on the result screen before the next game (esp. a sudden task win)
INTRO_LINGER = (
    5.5  # the opening card (frozen outpost, "something came in from the cold") before play
)
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
        self.revealed = False  # during ejection: has the human/Cold reveal happened yet?
        self.whisper = None  # [from, to] while a private DM indicator is flashing
        self.meeting = None
        self.task_q: asyncio.Queue = asyncio.Queue()  # engine task events → mirrored onto Artel
        self.task_pool: dict = {}  # room -> [open Artel task ids waiting to be claimed]
        self.task_working: dict = {}  # agent id -> the Artel task id it's currently doing
        self.task_room: dict = {}  # Artel task id -> room (to map the board back to rooms)
        self.inbox: dict = {}  # agent id -> [(from_name, text)] task-phase whispers received
        self.interrupted: set = set()  # agent ids that got a whisper → re-decide next tick
        self.deciding = 0  # agents that asked the LLM for an action this tick (for the ops board)
        self.feed: list = []  # rolling log of real Artel events (messages, claims, completes)
        self.feed_seq = 0  # monotonic id so the client appends only what's new
        self._restore_state()
        self._new_game()

    def push_feed(self, kind: str, **data) -> None:
        self.feed_seq += 1
        self.feed.append({"seq": self.feed_seq, "kind": kind, **data})
        del self.feed[:-60]

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
        self.task_room.clear()
        self.inbox.clear()
        self.interrupted.clear()
        self.feed.clear()
        while not self.task_q.empty():
            try:
                self.task_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.enqueue_task_events()

    def _new_game(self) -> None:
        self.g = new_game(_rng.randint(1, 2**31 - 1), n=N_AGENTS, impostors=N_IMPOSTORS)
        self.tasks_total = self.g.tasks_goal
        self.phase = "intro"  # opening card first; the loop holds it, then play begins
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
        if self.phase == "intro":
            return "the outpost wakes — something came in with the storm"
        if self.phase == "gameover":
            return f"{g.winner} won by {g.win_by} · {len(g.living())} left standing"
        if self.meeting is not None and self.phase in ("meeting", "vote", "ejection"):
            if self.meeting.victim is not None:
                return f"meeting: {g.by_id(self.meeting.victim).name} found dead in {self.meeting.room}"
            return f"emergency meeting in {self.meeting.room}"
        return f"task phase · {len(g.living())}/{len(g.agents)} alive · tick {g.tick}"

    def snapshot(self) -> dict:
        g = self.g
        # the Cold is unmasked only at the dramatic reveal (after the walk-out) or on the final board
        reveal = (self.phase == "ejection" and self.revealed) or self.phase == "gameover"
        agents = []
        for a in g.agents:
            d = {
                "id": a.id,
                "name": a.name,
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
                "transcript": [{"name": g.by_id(s).name, "text": t} for s, t in mt.transcript],
                "votes": {},
                "ejected": None,
                "ejected_was_thing": None,
                "whisper": self.whisper,  # [from, to] while a private DM is passing
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
            "feed": self.feed[-40:],  # live Artel event ticker
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
    G.whisper = None
    G.meeting = mt
    await _broadcast()

    async def on_item(kind: str, agent_id: int, payload) -> None:
        # paces the meeting: spoken lines hold on screen, private whispers flash a discreet indicator,
        # the table SETTLES before the vote, then votes land one at a time. Public lines + votes go on
        # the Artel bus; whispers are real private Artel DMs.
        if kind == "settle":  # discussion over → a beat before the vote
            G.phase = "vote"
            await _broadcast()
            await asyncio.sleep(PRE_VOTE)
            return
        name = G.g.by_id(agent_id).name
        if (
            kind == "whisper"
        ):  # a private DM — viewer only sees that a whisper passed between two seats
            G.phase = "meeting"
            await artel.dm(agent_id, payload["to"], payload["text"])
            G.whisper = [name, G.g.by_id(payload["to"]).name]
            G.push_feed("msg", frm=name, to=G.g.by_id(payload["to"]).name, text=payload["text"])
            await _broadcast()
            await asyncio.sleep(WHISPER_DELAY)
            G.whisper = None
            return
        if kind == "statement":
            G.phase = "meeting"
            await artel.say(agent_id, name, payload)
            G.push_feed("msg", frm=name, to=None, text=payload)
            await _broadcast()
            await asyncio.sleep(STMT_DELAY)
        else:
            G.phase = "vote"
            target = G.g.by_id(payload).name if payload is not None and payload >= 0 else "abstains"
            await artel.say(agent_id, name, f"votes {target}.", subject="alibi-vote")
            G.push_feed("msg", frm=name, to=None, text=f"votes {target}")
            await _broadcast()
            await asyncio.sleep(VOTE_DELAY)

    if llm.enabled():
        votes = await run_llm_meeting(G.g, mt, on_item, watched=lambda: bool(G.viewers))
    else:  # no key configured → canned statements + deterministic decider so the scene still plays
        from .brain import make_decider

        votes = await run_canned_meeting(G.g, mt, make_decider(share=True), on_item)
    G.g.apply_votes(mt, votes)
    # ejection: walk the ejected out (suspense), THEN reveal whether they were the Cold
    G.phase = "ejection"
    G.revealed = False
    await _broadcast()
    if mt.ejected is not None and G.viewers:  # skip the suspense beats if nobody's watching
        await asyncio.sleep(EJECT_WALK)
    G.revealed = True  # the storm-door reveal
    await _broadcast()
    if G.viewers:
        await asyncio.sleep(EJECT_REVEAL)
    if G.g.winner is None:  # meeting's over and the game goes on → back to the station
        G.phase = "task"
        await _broadcast()


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


def _task_title(room: str) -> str:
    return f"{_TASK_VERBS[hash(room) % len(_TASK_VERBS)]} the {room}"


def _target_id(g, name):
    a = next((o for o in g.living() if o.name == name), None)
    return a.id if a else None


async def _apply_action(g, a, action) -> str | None:
    # turn one tool call into engine intent + real Artel side-effects. Returns "meeting" if the agent
    # called one. Unknown/garbled actions are no-ops — the agent simply re-decides next tick.
    name = action.get("name")
    args = action.get("args") or {}
    if name == "go_to_task":
        room = args.get("room")
        if room in g.open_tasks:
            pool = G.task_pool.get(room) or []
            tid = pool.pop(0) if pool else None
            won = (await artel.claim_task(a.id, tid)) if (artel.enabled() and tid) else True
            if won and g.claim_room(a, room):
                if tid:
                    G.task_working[a.id] = tid
                G.push_feed("task", action="claimed", who=a.name, room=room)
            if tid:
                G.task_room.pop(tid, None)
    elif name == "follow":
        tid = _target_id(g, args.get("who"))
        if tid is not None:
            g.set_follow(a, tid)
    elif name == "move_to":
        g.set_goto(a, args.get("room", ""))
    elif name == "whisper":
        to = _target_id(g, args.get("who"))
        text = str(args.get("message", "")).strip()
        if to is not None and text:
            await artel.dm(a.id, to, f"{a.name}: {text}"[:280])
            G.inbox.setdefault(to, []).append((a.name, text[:120]))
            G.interrupted.add(to)
            G.whisper = [a.name, g.by_id(to).name]
            G.push_feed("msg", frm=a.name, to=g.by_id(to).name, text=text[:120])
    elif name == "eliminate":
        vid = _target_id(g, args.get("who"))
        if vid is not None and g.do_kill(a, vid):
            # any survivor who saw it re-decides next tick — they'll likely sound the alarm
            G.interrupted.update(w.id for w in g.living() if a.id in w.witnessed)
    elif name == "call_meeting":
        return "meeting"
    return None


async def _autonomous_tick():
    # the agents DECIDE (LLM tool calls), the engine EXECUTES. Free or just-whispered agents each pick one
    # action; committed agents (walking/working) keep going. Claims hit Artel for real contention; a body
    # or an agent-called alarm opens a meeting.
    g = G.g
    G.whisper = None
    meeting_caller = None
    deciders = [
        a for a in g.living() if g.needs_decision(a) or a.id in G.interrupted or g.prime_kill(a)
    ]
    G.interrupted.clear()
    G.deciding = len(deciders)
    if (
        deciders and not G.viewers
    ):  # audience left since the loop's gate check → don't spend this tick
        return None
    if deciders:
        reqs = [autonomy.build_request(g, a, G.inbox.get(a.id)) for a in deciders]
        for a in deciders:
            G.inbox.pop(a.id, None)  # whispers are consumed into this decision
        actions = await llm.act_many(reqs)
        for a, action in zip(deciders, actions):
            act = action if action is not None else g.default_action(a)
            if await _apply_action(g, a, act) == "meeting":
                meeting_caller = a
    mt = g.execute()
    for (
        ev
    ) in g.events:  # spawns → create Artel tasks, completes → complete them (claims done inline)
        try:
            if ev[0] == "spawn":
                room = ev[1]
                tid = await artel.create_task(_task_title(room))
                if tid:
                    G.task_pool.setdefault(room, []).append(tid)
                    G.task_room[tid] = room
            elif ev[0] == "complete":
                tid = G.task_working.pop(ev[1], None)
                if tid:
                    await artel.complete_task(ev[1], tid)
                done = G.g.by_id(ev[1])
                G.push_feed("task", action="completed", who=done.name, room=done.room)
        except Exception as e:
            log.warning("task mirror failed: %s", e)
    if mt is None and meeting_caller is not None:
        # if a body was just found (this meeting is a report, not a hunch), open it ON the corpse so the
        # client zooms to the scene of the crime before cutting to the table
        lk = g.last_kill
        if lk and lk["tick"] >= g.tick - 3 and not g.by_id(lk["victim"]).alive:
            mt = Meeting(g.tick, meeting_caller.id, lk["room"], lk["victim"])
        else:
            mt = Meeting(g.tick, meeting_caller.id, HUB, None)
    return mt


async def _game_loop():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        if not G.paused and G.viewers:
            if (
                G.phase == "intro"
            ):  # play the opening card, then begin (no ticks, no LLM during intro)
                await _broadcast()
                await asyncio.sleep(INTRO_LINGER)
                G.phase = "task"
                await _broadcast()
                continue
            async with G.lock:
                if llm.enabled():
                    mt = await _autonomous_tick()  # full-autonomous: agents drive the task phase
                else:
                    mt = G.g.step()  # offline / unprovisioned: the deterministic engine drives
                    G.enqueue_task_events()
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
            if G.g.tick >= MAX_TICKS:  # safety: stalemate → Cold wins
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
        "model": llm.POOL_DESC,
        "router": llm.metrics(),
        "deciding": G.deciding,
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
