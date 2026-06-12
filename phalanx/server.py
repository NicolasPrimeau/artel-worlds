from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
from pathlib import Path
from random import SystemRandom

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import os

from .agent import Squad
from .arena import Arena
from .config import DEFAULT
from .control import STRATEGIES, Bot

_rng = SystemRandom()
log = logging.getLogger("phalanx")


def _state_path() -> Path:
    # the running tally lives on a Fly volume so a DEPLOY no longer zeroes the series; the
    # in-flight match and all agent working state stay ephemeral — only the lasting record
    # (scoreboard, match counter, history, spend) survives. Wiping is an intentional act now.
    p = Path(os.environ.get("PHALANX_STATE", "/data/phalanx_state.json"))
    if not p.parent.exists():
        return Path(p.name)
    return p


STATIC = Path(__file__).parent / "static"
MATCH_END_LINGER = 4.0  # seconds to hold on the final positions before starting the next match
TICK_INTERVAL = 1.5  # MINIMUM seconds per tick — the commander runs async, so the tick clock never waits on a model
# (one paid decision per agent per tick, so a slower tick means proportionally fewer calls).
# It does NOT add think-time: the tick already waits for each model's answer (up to LLM_TIMEOUT),
# so reasoning depth is governed by max_tokens / MAX_TOOL_ROUNDS, not this floor.
# The whole demo: same arena, same guns. Artel is three real LLM agents — one per tank —
# coordinating live over artel.run; Red is deterministic seek-and-destroy bots that can't
# talk to each other. The only thing Artel has that Red doesn't is each other, through
# Artel. Nothing about Artel's play is scripted: the models drive every move.
COORDINATED = {"artel"}
# PHALANX_RED=llm turns the match into the REAL ablation demo: red is driven by the same
# LLM mind as blue with every Artel channel removed — Artel becomes the only variable.
RED_LLM = os.environ.get("PHALANX_RED", "bots").lower() == "llm"


class Phalanx:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.tokens: dict[str, str] = {}  # team -> secret
        self.scores: dict[str, int] = {}  # wins, accumulated across matches
        self.squad = Squad()  # the live Artel LLM agents (no-op until keys + a viewer)
        self.red_squad = Squad(solo=True, label="red") if RED_LLM else None
        self._squad_match = -1  # which match the squad is currently driving
        self._squad_started = -1  # which match the squad has already run on_start for
        self.history: list[dict] = []  # last matches' outcomes + kill logs, for /debug
        self.completed = 0  # matches that actually finished — match_no also counts restarts
        self.spend_days: dict[str, float] = {}  # iso day -> usd, the cost-per-day ledger
        self._ledgered = 0.0  # spend already attributed to a day
        self._restore_state()
        self._new_match()

    def _restore_state(self) -> None:
        try:
            raw = json.loads(_state_path().read_text())
        except Exception:
            return
        self.scores = {k: int(v) for k, v in (raw.get("scores") or {}).items()}
        self.history = list(raw.get("history") or [])[-10:]
        self.match_no = int(raw.get("match_no", -1))
        self.completed = int(raw.get("completed", 0))
        self.squad.spent = float(raw.get("squad_spent", 0.0))
        if self.red_squad is not None:
            self.red_squad.spent = float(raw.get("red_spent", 0.0))
        self.spend_days = {k: float(v) for k, v in (raw.get("spend_days") or {}).items()}
        self._ledgered = self.squad.spent + (self.red_squad.spent if self.red_squad else 0.0)

    def ledger_spend(self) -> None:
        import datetime

        total = self.squad.spent + (self.red_squad.spent if self.red_squad else 0.0)
        delta = total - self._ledgered
        if delta <= 0:
            return
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        self.spend_days[day] = round(self.spend_days.get(day, 0.0) + delta, 6)
        self._ledgered = total
        for k in sorted(self.spend_days)[:-30]:
            del self.spend_days[k]

    def persist_state(self) -> None:
        try:
            path = _state_path()
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "scores": self.scores,
                        "history": self.history,
                        "match_no": self.match_no,
                        "completed": self.completed,
                        "squad_spent": round(self.squad.spent, 6),
                        "red_spent": round(self.red_squad.spent, 6) if self.red_squad else 0.0,
                        "spend_days": self.spend_days,
                    }
                )
            )
            tmp.replace(path)
        except Exception as e:
            log.warning("phalanx state persist failed: %s", e)

    def _new_match(self) -> None:
        self.match_no = getattr(self, "match_no", -1) + 1
        self.seed = _rng.randint(1, 2**31 - 1)  # fresh random map layout each match
        seed = self.seed
        self.arena = Arena(DEFAULT, seed=seed)
        self.arena.seed_house(flip=bool(self.match_no % 2))  # alternate corners, fair series
        self.tokens.clear()
        self.squad.stop()
        self._squad_match = -1
        # one solo Bot per tank — each keeps its own private memory and never shares it, and
        # each team fields one of every temperament (brawler / ranger / opportunist), so the
        # solo side is three different hunters rather than one policy times three. The only
        # coordination in Phalanx is the Artel LLM squad talking over artel.run; Artel falls
        # back to these solo bots only when the squad is off (no keys / over the spend cap).
        nth: dict[str, int] = {}
        self.bots = {}
        for t in self.arena.tanks.values():
            i = nth.get(t.team, 0)
            nth[t.team] = i + 1
            self.bots[t.id] = Bot(t.id, t.team, STRATEGIES[i % len(STRATEGIES)])

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
            "map_radius": a.cfg.map_radius,
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
                    "target": t.target,
                    "energy": round(t.energy),
                    "team": t.team,
                    "player": self.is_player_team(t.team),
                    "coord": t.team in COORDINATED,
                }
                for t in a.tanks.values()
            ],
            "tracer_list": [
                {
                    "q": s["q"],
                    "r": s["r"],
                    "tq": s["tq"],
                    "tr": s["tr"],
                    "by": s.get("by"),
                    "tgid": s.get("tgid"),
                    "team": s["team"],
                    "path": s.get("path", []),
                    "dmg": s.get("dmg", 0),
                    "reward": s.get("reward", 0),
                    "power": s.get("power", 2),
                    "kind": s.get("kind", "hit"),
                }
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
    power: int = 2  # shot power 1-3: range 3/5/7 hexes for 1/2/4 energy


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


def _red_alive_ids() -> list[int]:
    return sorted(t.id for t in G.arena.tanks.values() if t.team not in COORDINATED)


def _manage_squad() -> None:
    # the live LLM agents only drive while someone's watching and the spend cap allows — no
    # one watching, no spend. Each new match binds them to that match's Artel tanks.
    if not G.viewers or not G.squad.enabled:
        if G._squad_match != -1:
            G.squad.stop()
            G._squad_match = -1
        return
    if G._squad_match != G.match_no:
        ids = _artel_alive_ids()
        G.squad.assign(ids)
        if G.red_squad is not None:
            G.red_squad.assign(_red_alive_ids())
        G._squad_match = G.match_no
        log.info("phalanx squad driving artel tanks %s with %s", ids, G.squad.status()["model"])


async def _tick_loop():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        if bool(G.viewers) or G.has_players():
            async with G.lock:
                a = G.arena
                _manage_squad()
                live_artel = G.squad.enabled and G._squad_match == G.match_no
                if live_artel and G._squad_started != G.match_no:
                    await G.squad.on_start()
                    if G.red_squad is not None:
                        await G.red_squad.on_start()
                    G._squad_started = G.match_no
                # collect every side's move for this tick: deterministic bots answer instantly;
                # each live Artel tank's LLM agent is asked now and awaited below.
                pending_llm: dict[int, asyncio.Task] = {}
                for t in list(a.tanks.values()):
                    if t.id in a.pending or G.is_player_team(t.team):
                        continue
                    if t.team in COORDINATED and live_artel:
                        pending_llm[t.id] = asyncio.create_task(G.squad.act(t.id, a.perceive))
                        continue
                    if G.red_squad is not None and live_artel and G.red_squad.enabled:
                        pending_llm[t.id] = asyncio.create_task(G.red_squad.act(t.id, a.perceive))
                        continue
                    p = a.perceive(t.id)
                    bot = G.bots.get(t.id)
                    if p and bot:
                        a.submit(t.id, bot.decide(p, a.cfg, a.tick_count))
                # the tick waits for all sides before resolving — a slow or failed agent just
                # leaves its tank on the arena's default (hold) for this tick.
                for tid, task in pending_llm.items():
                    intent = await task
                    if isinstance(intent, dict):
                        a.submit(tid, intent)
                artel_before = {t.id for t in a.tanks.values() if t.team in COORDINATED}
                a.step()
                artel_after = {t.id for t in a.tanks.values() if t.team in COORDINATED}
                ended = a.winner is not None or len(a.teams_alive()) <= 1
                if live_artel and not ended and artel_after < artel_before:
                    # a blue tank just fell — the opening plan is stale; the surviving lead
                    # calls an adjusted one (broadcast over Artel, like everything else)
                    asyncio.create_task(G.squad.on_loss(artel_after))
                if ended:
                    G.completed += 1
                    if a.winner:
                        G.scores[a.winner] = G.scores.get(a.winner, 0) + 1
                    # match record for /debug: outcome + the real kill log + how much the
                    # squad actually used Artel — enough to diagnose a losing streak after
                    # the fact without ssh, logs, or having watched the match live
                    G.history.append(
                        {
                            "match": G.match_no,
                            "seed": G.seed,
                            "winner": a.winner,
                            "ticks": a.tick_count,
                            "live_artel": live_artel,
                            "kills": list(a.events),
                            "artel_tools": dict(G.squad.tool_counts) if live_artel else {},
                            "squad_spent_usd": round(G.squad.spent, 4) if live_artel else None,
                        }
                    )
                    del G.history[:-10]
                    G.ledger_spend()
                    G.persist_state()
                    if live_artel:
                        survivors = {t.id for t in a.tanks.values() if t.team in COORDINATED}
                        asyncio.create_task(
                            G.squad.on_end(
                                a.winner == "artel",
                                survivors,
                                G.squad.current_assignment(),
                                "; ".join(a.events[-12:]),
                            )
                        )
                snap = G.snapshot()
            await _broadcast(snap)
            if ended:
                # hold on the final positions so viewers see who won before the next match
                await asyncio.sleep(MATCH_END_LINGER)
                async with G.lock:
                    G._new_match()
        await asyncio.sleep(max(0.0, TICK_INTERVAL - (loop.time() - start)))


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    tick = asyncio.create_task(_tick_loop())
    try:
        yield
    finally:
        tick.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await tick
        G.persist_state()
        await G.squad.aclose()


app = FastAPI(title="Phalanx — Artel Worlds", lifespan=_lifespan)


@app.get("/debug")
async def debug():
    # squad + match health, so a failing LLM is visible without ssh or reading logs
    a = G.arena
    return {
        "tick": a.tick_count,
        "match": G.match_no,
        "viewers": len(G.viewers),
        "live_artel": G.squad.enabled and G._squad_match == G.match_no,
        "red_mode": "llm-solo" if G.red_squad is not None else "bots",
        "scores": dict(G.scores),
        "completed": G.completed,
        "spend_days": dict(G.spend_days),
        "squad": G.squad.status(),
        "red_squad": G.red_squad.status() if G.red_squad is not None else None,
        "team_counts": a.stats()["team_counts"],
        "current_kills": list(a.events),
        "history": G.history[::-1],  # most recent match first
    }


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
    # the operator reset wipes the SERIES — scoreboard, history, match counter — and starts
    # fresh. Spend survives: it is money actually spent, and zeroing it would falsify the
    # cost reporting. (A mid-match restart is just a deploy; this is the intentional wipe.)
    async with G.lock:
        G.scores = {}
        G.history = []
        G.completed = 0
        G.match_no = -1
        G._new_match()
        G.persist_state()
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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC / "favicon.ico")


@app.get("/og.png", include_in_schema=False)
async def og_image():
    return FileResponse(STATIC / "og.png")


@app.get("/")
async def root():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"world": "Phalanx", "ui": "static/index.html not built yet"}
