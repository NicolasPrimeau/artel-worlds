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

from . import artel, autonomy, env, llm
from .engine import MAX_TICKS, new_game
from .meeting import (
    run_canned_meeting,
    run_llm_meeting,
)

# Verglas runs one game after another, but ONLY while someone is watching (free-tier Groq, like phalanx):
# no viewers → no ticks, no LLM calls. A game is a task phase (agents wander the station, the Cold
# kills) punctuated by meetings — which are streamed statement-by-statement so the chat builds live on
# the page. Crew win by clearing the task board or ejecting the Cold; the Cold wins by taking the last.

_rng = SystemRandom()
log = logging.getLogger("verglas")

STATIC = Path(__file__).parent / "static"
TASK_TICK = float(env("TICK_INTERVAL", "4.2"))  # min seconds per task-phase tick
STMT_DELAY = float(env("STMT_DELAY", "2.6"))  # base seconds a spoken line holds (jittered per line)
STMT_DELAY_MIN = float(env("STMT_DELAY_MIN", "1.4"))  # floor so quick retorts never blink past
STMT_DELAY_MAX = float(env("STMT_DELAY_MAX", "4.0"))  # ceiling so a long line never stalls the room
PRE_VOTE = float(env("PRE_VOTE", "2.0"))  # the table settles before the vote opens
DISCO_HOLD = float(
    env("DISCO_HOLD", "7.0")
)  # hold for the client's "body found" beat (slow readable gather + a brief hold) before the talk starts
VOTE_DELAY = float(env("VOTE_DELAY", "1.3"))  # seconds between revealed votes
WHISPER_DELAY = 1.6  # how long a private-whisper indicator flashes before play moves on
EJECT_WALK = (
    4.0  # the ejected researcher is walked out into the storm — BEFORE we reveal what they were
)
EJECT_REVEAL = 4.0  # then hold on the human/Cold reveal
GAMEOVER_LINGER = 14.0  # hold on the result screen before the next game (esp. a sudden task win)
INTRO_LINGER = (
    12.0  # the opening card — three lines fade in at reading pace, then a brief hold before play
)
STORM_SECONDS = float(
    env("STORM_SECONDS", "240")
)  # base length of the night (meetings included) — shorter than before to keep games brisk
_ADMIN_TOKEN = os.environ.get("WORLDS_ADMIN_TOKEN", "")
N_AGENTS = int(env("AGENTS", "10"))
N_IMPOSTORS = int(env("IMPOSTORS", "2"))  # the most Colds a night can have
TWO_COLD_CHANCE = float(
    env("TWO_COLD_CHANCE", "0.25")
)  # most nights have one Cold; two is the rarer, harder one


def _state_path() -> Path:
    p = Path(env("STATE", "/data/alibi_state.json"))
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


class Verglas:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.scores = {"crew": 0, "cold": 0}  # games won by each side
        self.fame: dict = {}  # name -> {games, cold, coldWins, crewWins} across all games (the leaderboard)
        self.recent: list[dict] = []  # [{win: crew-won?}] for the ops dots
        self.completed = 0
        self.paused = False
        self.game_secs = (
            0.0  # real seconds the current game has been playing — the storm clock to dawn
        )
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
        imps = 2 if N_IMPOSTORS >= 2 and _rng.random() < TWO_COLD_CHANCE else min(N_IMPOSTORS, 1)
        self.g = new_game(_rng.randint(1, 2**31 - 1), n=N_AGENTS, impostors=imps)
        self.g.storm_by_ticks = (
            False  # the real-seconds dawn clock below owns the storm win, not ticks
        )
        self.g.integrity_on = True  # the station-integrity drain/blackout is a live-only mechanic
        self.tasks_total = self.g.tasks_goal
        self.game_secs = 0.0  # fresh storm clock for the new night
        self.phase = "intro"  # opening card first; the loop holds it, then play begins
        self.meeting = None

    def _restore_state(self) -> None:
        try:
            raw = json.loads(_state_path().read_text())
        except Exception:
            return
        sc = raw.get("scores", {})
        self.scores = {
            "crew": int(sc.get("crew", 0)),
            "cold": int(sc.get("cold", sc.get("thing", 0))),
        }
        self.fame = dict(raw.get("fame") or {})
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
                        "fame": self.fame,
                        "recent": self.recent[-12:],
                        "completed": self.completed,
                        "paused": self.paused,
                        "spend": llm.SPEND,
                    }
                )
            )
            tmp.replace(path)
        except Exception as e:
            log.warning("verglas state persist failed: %s", e)

    def record_result(self) -> None:
        g = self.g
        crew_won = g.winner == "crew"
        self.scores["crew" if crew_won else "cold"] += 1
        self.recent.append({"win": crew_won})
        del self.recent[:-12]
        self.completed += 1
        for a in g.agents:  # tally each named AI's record for the leaderboard
            f = self.fame.setdefault(a.name, {"games": 0, "cold": 0, "coldWins": 0, "crewWins": 0})
            f["games"] += 1
            if a.impostor:
                f["cold"] += 1
                if not crew_won:
                    f["coldWins"] += 1
            elif crew_won:
                f["crewWins"] += 1
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
                "gx": round(a.gx, 2),
                "gy": round(a.gy, 2),
            }
            if reveal:
                d["cold"] = a.impostor
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
                "ejected_was_cold": None,
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
                    meeting["ejected_was_cold"] = g.by_id(mt.ejected).impostor
        return {
            "phase": self.phase,
            "revealed": self.revealed,
            "hunting": g.hunting,
            "round": len(g.meetings) + 1,
            "tick": g.tick,
            "dark": sorted(
                g.dark
            ),  # rooms currently unlit — the client dims them and flags relight jobs
            "stormElapsed": round(
                self.game_secs, 1
            ),  # real seconds into the night → the client's smooth dawn bar
            "stormTotal": STORM_SECONDS,
            "integrity": round(
                g.integrity, 1
            ),  # station health 0-100 → HUD readout; 0 = blackout, crew lose
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


G = Verglas()


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


async def _wait_watched() -> None:
    # PAUSE (don't abandon) whatever's running while nobody is watching or the world is admin-paused —
    # the meeting/ejection freezes in place and resumes the instant a viewer returns.
    while not G.viewers or G.paused:
        await asyncio.sleep(0.4)


async def _release_tasks(aids) -> None:
    # release the Artel claims held by these seats back onto the open board — called when a seat is killed
    # mid-task and at every meeting (reconvene drops everyone's task), so the live board never carries a
    # claim the engine has already abandoned.
    for aid in aids:
        tid = G.task_working.pop(aid, None)
        if tid:
            await artel.unclaim_task(aid, tid)


def _jit(secs: float, frac: float = 0.3) -> float:
    # sequencing beats are jittered so the pacing never feels metronomic; lingers/fades stay precise
    return secs * _rng.uniform(1.0 - frac, 1.0 + frac)


def _stmt_hold(text: str) -> float:
    # STMT_DELAY sets the pace for a ~12-word line; scale to the actual length, jitter ±35% so the
    # table never feels metronomic, then clamp so a terse jab still registers and a long accusation
    # doesn't stall the room
    base = _jit(STMT_DELAY * (len(text.split()) / 12.0), 0.35)
    return max(STMT_DELAY_MIN, min(STMT_DELAY_MAX, base))


async def _run_meeting(mt) -> None:
    G.g.reconvene()  # everyone — task-workers included — downs tools and gathers at the Mess Hall table
    await _release_tasks(
        list(G.task_working)
    )  # …so their Artel claims go back to the open board too
    G.phase = "meeting"
    G.revealed = False
    G.whisper = None
    G.meeting = mt
    await _broadcast()
    await _wait_watched()
    if mt.victim is not None:  # let the client's "body found" beat play before the talk
        await asyncio.sleep(DISCO_HOLD)

    async def on_item(kind: str, agent_id: int, payload, model: str | None = None) -> None:
        # paces the meeting: spoken lines hold on screen, private whispers flash a discreet indicator,
        # the table SETTLES before the vote, then votes land one at a time. Public lines + votes go on
        # the Artel bus; whispers are real private Artel DMs. `model` = the LLM that produced this line,
        # surfaced as a tiny per-line tag in the station log.
        if kind == "settle":  # discussion over → a beat before the vote
            G.phase = "vote"
            await _broadcast()
            await asyncio.sleep(_jit(PRE_VOTE))
            return
        name = G.g.by_id(agent_id).name
        if (
            kind == "whisper"
        ):  # a private DM — viewer only sees that a whisper passed between two seats
            G.phase = "meeting"
            await artel.dm(agent_id, payload["to"], payload["text"])
            G.whisper = [name, G.g.by_id(payload["to"]).name]
            G.push_feed(
                "msg",
                frm=name,
                to=G.g.by_id(payload["to"]).name,
                text=payload["text"],
                model=model,
            )
            await _broadcast()
            await asyncio.sleep(_jit(WHISPER_DELAY))
            G.whisper = None
            return
        if kind == "statement":
            G.phase = "meeting"
            await artel.say(agent_id, name, payload)
            G.push_feed("msg", frm=name, to=None, text=payload, model=model)
            await _broadcast()
            await asyncio.sleep(_stmt_hold(payload))
        else:
            G.phase = "vote"
            target = G.g.by_id(payload).name if payload is not None and payload >= 0 else "abstains"
            await artel.say(agent_id, name, f"votes {target}.", subject="verglas-vote")
            G.push_feed("msg", frm=name, to=None, text=f"votes {target}", model=model)
            await _broadcast()
            await asyncio.sleep(_jit(VOTE_DELAY))

    if llm.enabled():
        votes = await run_llm_meeting(G.g, mt, on_item, gate=_wait_watched)
    else:  # no key configured → canned statements + deterministic decider so the scene still plays
        from .brain import make_decider

        votes = await run_canned_meeting(G.g, mt, make_decider(share=True), on_item)
    G.g.apply_votes(mt, votes)
    # ejection: walk the ejected out (suspense), THEN reveal whether they were the Cold
    G.phase = "ejection"
    G.revealed = False
    await _broadcast()
    await _wait_watched()  # hold the walk-out + reveal until someone's watching (pause, don't skip)
    if mt.ejected is not None:
        await asyncio.sleep(EJECT_WALK)
    G.revealed = True  # the storm-door reveal
    await _broadcast()
    await _wait_watched()
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
    if kind == "dark":  # a room went dark → a relight job on the Artel board
        room = ev[1]
        tid = await artel.create_task(f"Relight the {room}")
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


async def _apply_action(g, a, action, model: str | None = None) -> str | None:
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
            G.push_feed("msg", frm=a.name, to=g.by_id(to).name, text=text[:120], model=model)
    elif name == "darken":
        room = args.get("room")
        if room:
            g.sabotage(a, room)  # snuff the lights — anonymous; makes a kill spot + a relight job
    elif name == "eliminate":
        vid = _target_id(g, args.get("who"))
        if vid is not None and g.do_kill(a, vid):
            # any survivor who saw it re-decides next tick — they may flee, and the body opens a meeting
            # the moment a living crewmate walks in on it (g.execute); there is no manual alarm
            G.interrupted.update(w.id for w in g.living() if a.id in w.witnessed)
    return None


async def _autonomous_tick():
    # the agents DECIDE (LLM tool calls), the engine EXECUTES. Free or just-whispered agents each pick one
    # action; committed agents (walking/working) keep going. Claims hit Artel for real contention; a meeting
    # opens only when a body is found (g.execute), never on a hunch.
    g = G.g
    G.whisper = None
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
        actions = await llm.act_many_m(reqs)
        for a, (action, model) in zip(deciders, actions):
            act = action if action is not None else g.default_action(a)
            await _apply_action(g, a, act, model)
    # no crewmate stands idle while the station is dark: any living crew left with nothing committed this
    # tick (e.g. they only whispered, or the model stalled) is sent to relight the nearest dark room. This
    # keeps the board moving AND routes crew through the dark rooms where bodies lie, so kills get found.
    for a in g.living(impostor=False):
        if g.needs_decision(a) and g.open_tasks:
            await _apply_action(g, a, g.default_action(a))
    mt = g.execute()  # a meeting opens here ONLY when a body is found (no emergency button)
    for ev in (
        g.events
    ):  # darkened rooms → create relight tasks, relights → complete them (claims done inline)
        try:
            if ev[0] == "dark":
                room = ev[1]
                tid = await artel.create_task(f"Relight the {room}")
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
    await _release_tasks(
        [i for i in list(G.task_working) if not G.g.by_id(i).alive]
    )  # release Artel claims held by anyone killed this tick
    return mt


async def _game_loop():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        try:
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
                    if G.game_secs >= STORM_SECONDS and G.g.winner is None:
                        G.g.winner, G.g.win_by = (
                            "crew",
                            "storm",
                        )  # dawn — the crew outlasted the storm
                    if G.g.winner is None:
                        if llm.enabled():
                            mt = (
                                await _autonomous_tick()
                            )  # full-autonomous: agents drive the task phase
                        else:
                            mt = (
                                G.g.step()
                            )  # offline / unprovisioned: the deterministic engine drives
                            G.enqueue_task_events()
                        if mt is not None and G.g.winner is None:
                            await _run_meeting(mt)
                    if G.g.winner is not None:
                        G.phase = "gameover"
                        G.record_result()
                        await _broadcast()
                G.game_secs += (
                    loop.time() - start
                )  # the night advances by this iteration's real time (meetings too)
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
        except asyncio.CancelledError:
            raise  # shutdown — let the lifespan cancel us cleanly
        except Exception as e:
            # one bad tick (a flaky Artel call, a transient hiccup) must never kill the loop and freeze the
            # world — log it and keep ticking. The lock is released by its context manager.
            log.warning("game loop tick failed: %s", e)
        await asyncio.sleep(max(0.0, _jit(TASK_TICK) - (loop.time() - start)))


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


app = FastAPI(title="Verglas — Artel Worlds", lifespan=_lifespan)


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


@app.get("/fame.json", include_in_schema=False)
async def fame():
    # the leaderboard: every named AI's record across all games, for the standings popup
    rows = [
        {
            "name": n,
            "games": f.get("games", 0),
            "wins": f.get("crewWins", 0) + f.get("coldWins", 0),
            "cold": f.get("cold", 0),
            "coldWins": f.get("coldWins", 0),
        }
        for n, f in G.fame.items()
    ]
    rows.sort(key=lambda r: (-r["wins"], -r["games"], r["name"]))
    return JSONResponse({"rows": rows, "games": G.completed})


@app.get("/state")
async def state():
    # snapshot WITHOUT the lock — the game loop holds it across a multi-second meeting, and a poll for the
    # current frame shouldn't hang behind that (same reasoning as the stream's initial send).
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
        G.scores = {"crew": 0, "cold": 0}
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
    return {"world": "Verglas", "ui": "static/index.html not built yet"}
