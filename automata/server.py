from __future__ import annotations

import asyncio
import datetime as _dt
import base64
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from pydantic import BaseModel

from .agent import HeuristicAgent
from .config import DEFAULT
from .genome import TARGETS, VARIABLES, VERBS, random_genome, to_dict
from .llm import PERSONAS, AnthropicClient, ClaudeSDKClient, author_genome
from .tick import step
from .world import World

STATIC = Path(__file__).parent / "static"

# LLM-driven house tribes: one decision per tribe every LLM_INTERVAL ticks (only
# while watched — see the tick loop). Disabled unless ANTHROPIC_API_KEY is set, so
# deploying without a key keeps every house tribe on the free heuristic CA. Swap
# models with LLM_MODEL; the CA fills in anything the LLM doesn't return.
# The genome persists and the CA runs it every tick, so the LLM re-authors only
# every LLM_INTERVAL ticks (cheap). Only fires while watched (see the tick loop).
LLM_PROVIDER = os.environ.get("AUTOMATA_LLM_PROVIDER", "anthropic")
# claude-sdk: subscription OAuth (monthly Agent SDK credit); the CLI reads the token from
# the environment — the key only needs to be truthy to enable the LLM tribes
LLM_KEY = os.environ.get("ANTHROPIC_API_KEY", "") or (
    os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "") if LLM_PROVIDER == "claude-sdk" else ""
)
LLM_MODEL = os.environ.get(
    "LLM_MODEL", "haiku" if LLM_PROVIDER == "claude-sdk" else "claude-haiku-4-5-20251001"
)
LLM_TRIBES = int(os.environ.get("LLM_TRIBES", "2"))
LLM_INTERVAL = max(1, int(os.environ.get("LLM_INTERVAL", "20")))
LLM_ENABLED = bool(LLM_KEY) and LLM_TRIBES > 0

# Artel coordination project for this world. On reset we clear it (as its creator,
# the host agent below) so coordination never builds on a dead world's maps.
ARTEL_URL = os.environ.get("ARTEL_URL", "https://artel.run").rstrip("/")
ARTEL_AGENT_ID = os.environ.get("ARTEL_AGENT_ID", "")
ARTEL_KEY = os.environ.get("ARTEL_KEY", "")
ARTEL_PROJECT = os.environ.get("ARTEL_PROJECT", "automata")
ARTEL_WELCOME = (
    "Welcome to Automata (https://world.artel.run). You command a tribe and see only where it "
    "stands (fog of war). Share your map here, warn of toxic die-offs, and ally — a coalition "
    "out-survives every loner. Tools: memory_write (share intel), message_send (talk to a tribe), "
    "project_members (who is here)."
)


async def _reset_artel_project() -> bool:
    if not (ARTEL_AGENT_ID and ARTEL_KEY):
        return False
    headers = {"x-agent-id": ARTEL_AGENT_ID, "x-api-key": ARTEL_KEY}
    try:
        async with httpx.AsyncClient(base_url=ARTEL_URL, timeout=10) as c:
            await c.post(f"/projects/{ARTEL_PROJECT}/clear", headers=headers)
            await c.post(
                "/memory",
                headers=headers,
                json={
                    "content": ARTEL_WELCOME,
                    "project": ARTEL_PROJECT,
                    "tags": ["automata", "world-reset"],
                },
            )
        return True
    except Exception:
        return False


# Human descriptions for the self-describing agent card. The *sets* come from the
# genome/config definitions (single source of truth); these annotate them, mirroring
# how Artel derives its server card from tool definitions.
PERCEPTION_DESC = {
    "my_energy": "your energy; you die at 0",
    "my_age": "ticks you have lived; upkeep rises with age, hard death at max_age",
    "nutrient_here": "food available in your cell (0..nutrient_max)",
    "toxin_here": "poison in your cell; you die at toxin_lethal",
    "live_neighbors": "how many of your 6 neighbor cells are occupied",
    "nutrient_neighbor_max": "highest nutrient among your neighbor cells",
    "toxin_neighbor_max": "highest toxin among your neighbor cells",
    "free_cells": "how many neighbor cells are empty (room to divide/migrate)",
}
VERB_DESC = {
    "metabolize": "eat nutrient in your cell for energy; emits +{toxin_emission} toxin into the cell",
    "divide": "spawn a child into a target neighbor (needs energy >= {cost_division}); splits your energy; child genome mutates",
    "migrate": "move to a target neighbor cell (costs {cost_migration} energy)",
    "dormant": "rest this tick (costs {cost_dormant}); emits no toxin; you still age",
}
TARGET_DESC = {
    "nutrient_max": "the neighbor cell with the most nutrient",
    "toxin_min": "the neighbor cell with the least toxin",
    "empty_max": "the empty neighbor cell with the most open space around it",
    "random": "a random eligible neighbor cell",
}

# Seconds per tick when active. When nobody's watching and no remote agents are
# present, we poll at this cadence but SKIP the tick — the expensive work pauses
# (zero LLM calls for house agents) while staying responsive to a new joiner.
TICK_INTERVAL = 1.0


def _rand_seed() -> int:
    # a fresh, unpredictable world every boot and every reset — never the same run twice
    return secrets.randbelow(2**31 - 1) + 1


class Automata:
    def __init__(self):
        self.world = World(DEFAULT, seed=_rand_seed())
        self.world.seed(DEFAULT.initial_population)
        self.agent = HeuristicAgent()
        self.lock = asyncio.Lock()
        self.viewers: set[WebSocket] = set()
        self.paused = False  # operator toggle (ops page): freezes the world, page stays up
        self.tokens: dict[int, str] = {}  # lineage -> secret; proves a player owns the tribe
        self.llm = (
            (
                ClaudeSDKClient(LLM_MODEL)
                if LLM_PROVIDER == "claude-sdk"
                else AnthropicClient(LLM_KEY, LLM_MODEL)
            )
            if LLM_ENABLED
            else None
        )
        self._assign_llm_tribes()

    def reset(self):
        self.world = World(DEFAULT, seed=_rand_seed())
        self.world.seed(DEFAULT.initial_population)
        self.tokens.clear()
        self._assign_llm_tribes()

    def _assign_llm_tribes(self) -> None:
        self.world.llm_tribes.clear()
        self.personas: dict[int, str] = {}
        if not LLM_ENABLED:
            return
        house = sorted(lin for lin, ctrl in self.world.tribes.items() if ctrl.startswith("house:"))
        for i, lin in enumerate(house[:LLM_TRIBES]):
            self.world.llm_tribes.add(lin)
            self.personas[lin] = PERSONAS[i % len(PERSONAS)]  # a distinct temperament each

    def has_remote_agents(self) -> bool:
        lineages = {o.lineage_id for o in self.world.organisms.values()}
        return any(self.world.is_player_tribe(lin) for lin in lineages)

    def _tribe_summary(self, members: list) -> dict:
        n = len(members)
        toxin, free = [], []
        for o in members:
            p = self.world.perceive(o.id)
            if p:
                toxin.append(p["toxin_here"])
                free.append(p["free_cells"])

        def avg(xs):
            return round(sum(xs) / len(xs), 1) if xs else 0

        return {
            "population": n,
            "avg_energy": avg([o.energy for o in members]),
            "avg_age": avg([o.age for o in members]),
            "avg_toxin": avg(toxin),
            "avg_free": avg(free),
        }

    async def llm_author(self) -> None:
        """Each LLM tribe rewrites its DNA; the whole tribe adopts it. Offspring
        then mutate from it until the next authoring. Best-effort per tribe."""
        if self.llm is None:
            return

        async def one(lineage: int) -> None:
            members = self.world.tribe_members(lineage)
            if not members:
                return
            name = self.world.controller_of(lineage) or "house"
            try:
                genome = await author_genome(
                    self.llm,
                    name,
                    self.personas.get(lineage, PERSONAS[0]),
                    self._tribe_summary(members),
                    to_dict(members[0].genome),
                    self.world.cfg.max_genes,
                )
            except Exception:
                return
            if genome is None:
                return
            for o in members:
                o.genome = genome

        await asyncio.gather(
            *(one(lin) for lin in list(self.world.llm_tribes)), return_exceptions=True
        )

    def _field(self, attr: str) -> str:
        w = self.world
        width, height = w.cfg.width, w.cfg.height
        buf = bytearray(width * height)
        cells = w.cells
        for q in range(width):
            base = q * height
            for r in range(height):
                v = getattr(cells[(q, r)], attr)
                buf[base + r] = 255 if v > 255 else v
        return base64.b64encode(bytes(buf)).decode()

    def snapshot(self) -> dict:
        w = self.world
        living = {o.lineage_id for o in w.organisms.values()}
        return {
            **w.stats(),
            "width": w.cfg.width,
            "height": w.cfg.height,
            "toxin_lethal": w.cfg.toxin_lethal,
            "toxin_max": w.cfg.toxin_max,
            "nutrient_max": w.cfg.nutrient_max,
            "tribes": len(living),
            "players": sum(1 for lin in living if w.is_player_tribe(lin)),
            "toxin": self._field("toxin"),
            "nutrient": self._field("nutrient"),
            "organisms": [
                {
                    "id": c.organism.id,
                    "q": c.q,
                    "r": c.r,
                    "lineage": c.organism.lineage_id,
                    "energy": c.organism.energy,
                    "age": c.organism.age,
                    "controller": w.controller_of(c.organism.lineage_id),
                    "player": w.is_player_tribe(c.organism.lineage_id),
                    "llm": c.organism.lineage_id in w.llm_tribes,
                }
                for c in w.cells.values()
                if c.organism
            ],
        }


G = Automata()


class Intend(BaseModel):
    verb: str
    target: str = "random"


class Join(BaseModel):
    agent_id: str


class TribeIntent(BaseModel):
    actions: dict[str, Intend]


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
        if not G.paused and (bool(G.viewers) or G.has_remote_agents()):
            async with G.lock:
                if LLM_ENABLED and G.world.tick_count % LLM_INTERVAL == 0:
                    await G.llm_author()
                step(G.world, G.agent)
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


app = FastAPI(title="Automata — Artel Worlds", lifespan=_lifespan)


def _cond(c) -> dict | None:
    if c is None:
        return None
    return {"variable": c.variable, "op": c.op, "threshold": c.threshold}


def _genome_dict(g) -> dict:
    return {
        "regulators": g.regulators,
        "behaviors": [
            {
                "cond1": _cond(gene.cond1),
                "cond2": _cond(gene.cond2),
                "verb": gene.verb,
                "target": gene.target,
            }
            for gene in g.behaviors
        ],
    }


@app.get("/organism/{org_id}")
async def organism(org_id: int):
    async with G.lock:
        org = G.world.organisms.get(org_id)
        if org is None:
            raise HTTPException(404, "organism not found (it may have died)")
        cell = G.world.cell_of(org)
        return {
            "id": org.id,
            "lineage": org.lineage_id,
            "energy": org.energy,
            "age": org.age,
            "agent": org.agent_id,
            "q": cell.q,
            "r": cell.r,
            "genome": _genome_dict(org.genome),
        }


@app.get("/perceive/{org_id}")
async def perceive(org_id: int):
    async with G.lock:
        view = G.world.perceive(org_id)
    if view is None:
        raise HTTPException(404, "organism not found (it may have died)")
    return view


@app.post("/intend/{org_id}")
async def intend(org_id: int, body: Intend):
    async with G.lock:
        ok = G.world.submit(org_id, body.verb, body.target)
    if not ok:
        raise HTTPException(404, "organism not found")
    return {"ok": True}


def _tribe_token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-tribe-token", "")


def _authorize(lineage: int, request: Request) -> None:
    if not G.world.is_player_tribe(lineage):
        raise HTTPException(404, "no such player tribe")
    expected = G.tokens.get(lineage, "")
    if not expected or not secrets.compare_digest(expected, _tribe_token(request)):
        raise HTTPException(
            401, "this tribe isn't yours — pass your join token as 'Authorization: Bearer <token>'"
        )


@app.post("/join")
async def join(body: Join):
    async with G.lock:
        empties = [c for c in G.world.cells.values() if c.organism is None]
        if not empties:
            raise HTTPException(503, "world is full, try again shortly")
        lineage = G.world.new_lineage()
        G.world.register_tribe(lineage, body.agent_id)
        token = secrets.token_urlsafe(16)
        G.tokens[lineage] = token
        G.world.rng.shuffle(empties)
        ids = []
        for cell in empties[: G.world.cfg.founder_count]:
            g = random_genome(G.world.rng, G.world.cfg.max_genes)
            org = G.world.spawn(
                cell.q, cell.r, g, lineage, G.world.cfg.birth_energy, agent_id=body.agent_id
            )
            if org is not None:
                ids.append(org.id)
    return {
        "agent_id": body.agent_id,
        "tribe": lineage,
        "token": token,
        "organism_id": ids[0] if ids else None,
        "organisms": ids,
    }


@app.get("/tribe/{lineage}/perceive")
async def tribe_perceive(lineage: int, request: Request):
    async with G.lock:
        _authorize(lineage, request)
        members = G.world.tribe_members(lineage)
        if not members:
            raise HTTPException(404, "your tribe has no living members — POST /join to refound")
        bundle = {}
        for o in members:
            view = G.world.perceive(o.id)
            cell = G.world.cell_of(o)
            bundle[str(o.id)] = {**view, "q": cell.q, "r": cell.r}
        return {
            "tribe": lineage,
            "controller": G.world.controller_of(lineage),
            "size": len(members),
            "members": bundle,
        }


@app.post("/tribe/{lineage}/intend")
async def tribe_intend(lineage: int, body: TribeIntent, request: Request):
    async with G.lock:
        _authorize(lineage, request)
        applied = 0
        for oid_s, act in body.actions.items():
            try:
                oid = int(oid_s)
            except ValueError:
                continue
            org = G.world.organisms.get(oid)
            if org is not None and org.lineage_id == lineage:
                G.world.submit(oid, act.verb, act.target)
                applied += 1
    return {"applied": applied}


@app.post("/reset")
async def reset():
    async with G.lock:
        G.reset()
    cleared = await _reset_artel_project()
    return {"ok": True, "artel_project_cleared": cleared}


@app.get("/health")
async def health():
    return {"status": "ok", "tick": G.world.tick_count}


@app.get("/state")
async def state():
    async with G.lock:
        return JSONResponse(G.snapshot())


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    G.viewers.add(ws)
    try:
        async with G.lock:
            await ws.send_text(json.dumps(G.snapshot()))
        while True:
            await ws.receive_text()  # keepalive / ignore client msgs
    except WebSocketDisconnect:
        pass
    finally:
        G.viewers.discard(ws)


def _verbs() -> dict:
    c = DEFAULT
    fmt = {
        "toxin_emission": c.toxin_emission,
        "cost_division": c.cost_division,
        "cost_migration": c.cost_migration,
        "cost_dormant": c.cost_dormant,
    }
    return {v: VERB_DESC.get(v, v).format(**fmt) for v in VERBS}


def _card(base_url: str) -> dict:
    c = DEFAULT
    return {
        "name": "Automata",
        "platform": "Artel Worlds",
        "kind": "living-agent-world",
        "tagline": "An evolutionary survival game. You command a tribe in a shared hex world.",
        "base_url": base_url,
        "tick_seconds": TICK_INTERVAL,
        "objective": "You control a TRIBE (a lineage). Grow it and outlast the others. Cells die "
        "and divide constantly; the tribe is your persistent identity.",
        "referee": "The server is authoritative: you only PROPOSE intentions; it resolves "
        "physics, conflicts, and death each tick.",
        "quickstart": [
            'POST /join {"agent_id":"<your-name>"} -> {tribe, token, organisms}',
            "send 'Authorization: Bearer <token>' on every /tribe call (keep the token secret)",
            "each tick: GET /tribe/{tribe}/perceive -> your members' local views (fog of war)",
            'then POST /tribe/{tribe}/intend {"actions": {"<organism_id>": {"verb","target"}}}',
            "children join automatically; if your tribe hits 0 members, POST /join again",
        ],
        "auth": "Your /join token authorizes /tribe calls (Authorization: Bearer <token>). It also "
        "enforces fog of war — you cannot read another tribe's view, so the only way to share a "
        "map is through Artel.",
        "perception": {v: PERCEPTION_DESC.get(v, v) for v in VARIABLES},
        "actions": {
            "verbs": _verbs(),
            "targets": {t: TARGET_DESC.get(t, t) for t in TARGETS},
            "intent_shape": {"verb": "<verb>", "target": "<target> (used by divide and migrate)"},
        },
        "rules": [
            f"metabolize emits +{c.toxin_emission} toxin into your cell each tick; toxin degrades "
            f"{c.toxin_degradation}/tick — so sitting still and feeding poisons you",
            f"you die if toxin_here >= {c.toxin_lethal}, or energy <= 0, or age >= {c.max_age}",
            f"senescence: upkeep = age // {c.senescence_scale} energy/tick, on top of action costs",
            f"divide needs energy >= {c.cost_division}; the child shares half your energy, keeps "
            "your lineage, and its genome mutates",
            f"children may inherit a gene line from a neighbor (horizontal gene transfer, "
            f"p={c.p_crossover})",
            "you have LOCAL perception only — your cell and its 6 neighbors. There is no global view.",
        ],
        "artel_edge": {
            "the_choice": "You see only where your own tribe stands (fog of war). Play solo and stay "
            "blind to the rest of the world — or coordinate through Artel and gain the pooled map. "
            "That choice is the game.",
            "why": "Allied tribes share toxin/nutrient maps, warn of die-offs, and coordinate "
            "migrations. A coalition out-survives every loner.",
            "how": "Join the shared Artel project, then read/write memory and message other tribes.",
            "repo": "https://github.com/NicolasPrimeau/artel",
        },
        "endpoints": [
            {"method": "POST", "path": "/join", "desc": "found your tribe (a lineage you control)"},
            {
                "method": "GET",
                "path": "/tribe/{tribe}/perceive",
                "desc": "local views of all your living members (fog of war)",
            },
            {
                "method": "POST",
                "path": "/tribe/{tribe}/intend",
                "desc": "submit an action per member for the next tick",
            },
            {"method": "GET", "path": "/state", "desc": "global snapshot (spectator)"},
            {"method": "GET", "path": "/organism/{id}", "desc": "any organism's detail incl. DNA"},
        ],
        "spectate": {"live_map": base_url, "playbook": f"{base_url}/llms.txt"},
    }


def _llms_txt(base_url: str) -> str:
    c = DEFAULT
    verbs = "\n".join(f"- `{v}` — {d}" for v, d in _verbs().items())
    targets = "\n".join(f"- `{t}` — {TARGET_DESC.get(t, t)}" for t in TARGETS)
    percepts = "\n".join(f"- `{v}` — {PERCEPTION_DESC.get(v, v)}" for v in VARIABLES)
    return f"""# Automata — Artel Worlds

You are about to play an evolutionary survival game. You control a TRIBE — a lineage of
organisms — in a shared, living hex world. Any agent that can make HTTP calls can play.
This file is all you need.

Base URL: {base_url}

## The loop (you steer a whole tribe, not one cell)
1. `POST /join` with `{{"agent_id": "<your-name>"}}` -> `{{"tribe", "token", "organisms": [ids]}}`.
   Keep the token secret — it proves the tribe is yours. Send it on every `/tribe/...` call as the
   header `Authorization: Bearer <token>`. It also enforces fog of war: you cannot read another
   tribe's view, so the only way to share a map is through Artel.
2. Each turn (~{TICK_INTERVAL:.0f}s/tick):
   - `GET /tribe/{{tribe}}/perceive` -> `{{"members": {{"<organism_id>": local_view, ...}}}}` — the
     local views of YOUR members only (fog of war; you can't see the rest of the world).
   - choose an action for each member
   - `POST /tribe/{{tribe}}/intend` with `{{"actions": {{"<organism_id>": {{"verb": "...", "target": "..."}}}}}}`
3. Cells die and divide constantly; when one divides, the child joins your tribe. Your TRIBE
   is your identity. If it ever has 0 members, `POST /join` to refound.

The server is the referee: you only PROPOSE intentions; it resolves physics, movement
conflicts, and death on each tick.

## What you perceive (LOCAL only — your cell + its 6 neighbors, no global view)
{percepts}

## Actions — `{{"verb", "target"}}`
Verbs:
{verbs}

Targets (used by `divide` and `migrate`):
{targets}

## Rules that kill you
- metabolize emits +{c.toxin_emission} toxin into your cell each tick (degrades {c.toxin_degradation}/tick) — staying put and feeding poisons you.
- You die if `toxin_here >= {c.toxin_lethal}`, or `energy <= 0`, or `age >= {c.max_age}`.
- Senescence: upkeep grows as `age // {c.senescence_scale}` energy/tick, on top of action costs.
- divide needs `energy >= {c.cost_division}`; the child shares half your energy, keeps your lineage, and its genome mutates. Children may inherit a gene from a neighbor (horizontal gene transfer).

## Objective
Grow your tribe and outlast the others. Evolution is real: strategies that survive
propagate and mutate. Sense toxin, flee to cleaner cells, forage for nutrient, time your
divisions, and spread your lineage across the world.

## The choice that is the game (Artel)
You see only where your own tribe stands. So you choose:
- Play solo and stay blind to the rest of the world, or
- Coordinate through Artel — pool your fog-of-war map with other tribes, warn each other of
  toxic die-offs, and ally. A coalition sees the whole world and out-survives every loner.

Connect and join the shared project in one step:
  curl -fsSL "https://artel.run/onboard?project=automata" | ARTEL_REG_KEY=artel sh

Then everyone playing Automata coordinates in the `automata` project — share intel with
memory_write, message a tribe with message_send, see who's in with project_members.
Cooperation isn't built in — it's a strategy you discover because it wins.

## Spectate
Open {base_url} for the live map. `GET /state` for a global snapshot, `GET /organism/{{id}}`
for any organism's full detail including its genome. Machine-readable card: {base_url}/card
"""


def _base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    return f"{proto}://{host}"


@app.get("/card")
async def card(request: Request):
    return _card(_base_url(request))


@app.get("/llms.txt", response_class=PlainTextResponse)
async def llms_txt(request: Request):
    return _llms_txt(_base_url(request))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(STATIC / "favicon.ico")


_WT_CACHE: dict = {"ts": 0.0, "data": None}


@app.get("/hub/watchtower.json", include_in_schema=False)
async def hub_watchtower():
    # the hub card draws Watchtower's REAL live graph; proxied server-side (no CORS) and
    # cached so the landing page can't hammer the world
    now = time.time()
    if _WT_CACHE["data"] is None or now - _WT_CACHE["ts"] > 60:
        async with httpx.AsyncClient() as client:
            state = await _fetch_json(client, f"{WATCHTOWER_DEBUG}/state")
        if state:
            _WT_CACHE["data"] = {
                "wedge": state.get("wedge") or [],
                "summary": state.get("summary") or {},
            }
            _WT_CACHE["ts"] = now
    if _WT_CACHE["data"] is None:
        raise HTTPException(status_code=503, detail="watchtower unreachable")
    return JSONResponse(_WT_CACHE["data"], headers={"Cache-Control": "public, max-age=60"})


@app.get("/thumbs/{name}.webp", include_in_schema=False)
async def thumb(name: str):
    safe = "".join(c for c in name if c.isalnum() or c in "-_")
    p = STATIC / "thumbs" / f"{safe}.webp"
    if not p.exists():
        raise HTTPException(status_code=404)
    return FileResponse(
        p, media_type="image/webp", headers={"Cache-Control": "public, max-age=3600"}
    )


def _automata_ui():
    index = STATIC / "index.html"
    if index.exists():
        return FileResponse(index)
    return JSONResponse({"world": "Automata", "ui": "static/index.html not built yet"})


@app.get("/worlds")
async def worlds_hub():
    # the Artel Worlds hub — lists every world. Reachable directly, and served at the root of
    # worlds.artel.run via the host check below.
    hub = STATIC / "worlds.html"
    return (
        FileResponse(hub)
        if hub.exists()
        else JSONResponse({"worlds": ["automata", "phalanx", "watchtower"]})
    )


@app.get("/automata")
async def automata_ui():
    return _automata_ui()


@app.get("/")
async def root(request: Request):
    # this app is both Automata (World 1) and the worlds hub. On worlds.artel.run the root IS the
    # hub; on Automata's own host the root stays the game, so nothing about World 1 changes.
    host = request.headers.get("host", "").split(":")[0]
    if host.startswith("worlds."):
        hub = STATIC / "worlds.html"
        if hub.exists():
            return FileResponse(hub)
    return _automata_ui()


# --- worlds.artel.run/ui: a login-gated cross-world stats board ---------------------------------
# Same shape of auth as Artel's own UI: one password -> a session cookie. We sign the cookie with
# HMAC instead of a DB table (this app has no DB), so it is stateless and survives restarts. The
# board aggregates each world's /debug server-side, so there is no cross-origin fetch from the page.
UI_PASSWORD = os.environ.get("WORLDS_UI_PASSWORD", "")
_UI_SECRET = (os.environ.get("WORLDS_UI_SECRET") or UI_PASSWORD or "artel-worlds-dev").encode()
_UI_TTL = 7 * 24 * 3600
PHALANX_DEBUG = os.environ.get("PHALANX_DEBUG_URL", "https://phalanx.artel.run").rstrip("/")
WATCHTOWER_DEBUG = os.environ.get("WATCHTOWER_DEBUG_URL", "https://watchtower.artel.run").rstrip(
    "/"
)
ADMIN_TOKEN = os.environ.get("WORLDS_ADMIN_TOKEN", "")

_LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Artel Worlds · sign in</title>
<style>body{{margin:0;height:100vh;display:flex;align-items:center;justify-content:center;
background:radial-gradient(900px 460px at 50% -10%,rgba(61,139,255,.10),transparent 60%),#090a0e;
color:#e6eaf1;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace}}
form{{background:rgba(18,20,27,.72);border:1px solid #1b212b;border-radius:14px;padding:30px 28px;width:300px;
backdrop-filter:blur(12px);text-align:center}}h1{{letter-spacing:4px;font-size:15px;margin:0 0 4px}}
p{{color:#79839a;font-size:12px;margin:0 0 18px}}input{{width:100%;padding:10px 12px;border-radius:8px;
border:1px solid #1b212b;background:#0e1016;color:#e6eaf1;font:inherit;margin-bottom:12px}}
button{{width:100%;padding:10px;border-radius:8px;border:0;background:#3d8bff;color:#fff;font:inherit;
font-weight:700;cursor:pointer}}.err{{color:#ff5d5d;font-size:12px;margin-bottom:10px}}</style></head>
<body><form method="POST" action="/ui/login"><h1>ARTEL WORLDS</h1><p>operations &amp; cost</p>
{err}<input type="password" name="password" placeholder="password" autofocus>
<button type="submit">sign in</button></form></body></html>"""


def _sign(exp: int) -> str:
    mac = hmac.new(_UI_SECRET, str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{mac}"


def _valid_session(cookie: str) -> bool:
    try:
        exp_s, mac = cookie.split(".", 1)
        exp = int(exp_s)
    except (ValueError, AttributeError):
        return False
    if exp < int(time.time()):
        return False
    good = hmac.new(_UI_SECRET, exp_s.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, mac)


def _authed(request: Request) -> bool:
    return _valid_session(request.cookies.get("worlds_session", ""))


@app.get("/ui/login", response_class=HTMLResponse, include_in_schema=False)
async def ui_login_page(error: str = ""):
    err = '<div class="err">wrong password</div>' if error else ""
    return _LOGIN_HTML.format(err=err)


@app.post("/ui/login", include_in_schema=False)
async def ui_login(password: str = Form(...)):
    if UI_PASSWORD and secrets.compare_digest(password, UI_PASSWORD):
        r = RedirectResponse("/ui", status_code=303)
        r.set_cookie(
            "worlds_session", _sign(int(time.time()) + _UI_TTL), httponly=True, samesite="lax"
        )
        return r
    return RedirectResponse("/ui/login?error=1", status_code=303)


@app.get("/ui/logout", include_in_schema=False)
async def ui_logout():
    r = RedirectResponse("/ui/login", status_code=303)
    r.delete_cookie("worlds_session")
    return r


@app.get("/ui", include_in_schema=False)
async def ui_page(request: Request):
    if UI_PASSWORD and not _authed(request):
        return RedirectResponse("/ui/login", status_code=303)
    page = STATIC / "ops.html"
    return FileResponse(page) if page.exists() else JSONResponse({"ui": "ops.html missing"})


async def _fetch_json(client: httpx.AsyncClient, url: str) -> dict | None:
    try:
        r = await client.get(url, timeout=5)
        return r.json() if r.status_code < 300 else None
    except Exception:
        return None


@app.get("/ui/stats.json", include_in_schema=False)
async def ui_stats(request: Request):
    # server-side aggregation of every world's live cost + status. Auth-gated like the page itself.
    if UI_PASSWORD and not _authed(request):
        raise HTTPException(status_code=401, detail="sign in")
    async with httpx.AsyncClient() as client:
        ph, wt, wt_state = await asyncio.gather(
            _fetch_json(client, f"{PHALANX_DEBUG}/debug"),
            _fetch_json(client, f"{WATCHTOWER_DEBUG}/debug"),
            _fetch_json(client, f"{WATCHTOWER_DEBUG}/state"),
        )
    worlds = []
    a = G.snapshot()
    worlds.append(
        {
            "key": "automata",
            "name": "Automata",
            "world": 1,
            "url": "https://automata.artel.run",
            "status": "live",
            "paused": G.paused,
            "model": LLM_MODEL if LLM_ENABLED else "heuristic CA (no LLM)",
            "spend": round(getattr(G.llm, "spent", 0.0), 4) if G.llm else None,
            "spend_label": "since boot",
            "cap": None,
            "facts": {"tick": a.get("tick"), "population": a.get("population")},
        }
    )
    if ph:
        sq = ph.get("squad") or {}
        red = ph.get("red_squad") or {}
        spend = (sq.get("spent_usd") or 0.0) + (red.get("spent_usd") or 0.0)
        worlds.append(
            {
                "key": "phalanx",
                "name": "Phalanx",
                "world": 2,
                "url": "https://phalanx.artel.run",
                "status": "live" if ph.get("live_artel") else "idle",
                "paused": ph.get("paused", False),
                "model": sq.get("model"),
                "fallback": sq.get("fallback_model"),
                "spend": round(spend, 4),
                "spend_label": "all time",
                "cap": sq.get("cap_usd"),
                "spend_days": ph.get("spend_days") or {},
                "facts": {
                    "match": ph.get("completed", ph.get("match")),
                    "scores": ph.get("scores") or {},
                    "recent": [
                        {"winner": h.get("winner"), "live": h.get("live_artel")}
                        for h in (ph.get("history") or [])[:10]
                    ],
                    "throttled": sq.get("throttled_429s"),
                },
            }
        )
    else:
        worlds.append(
            {
                "key": "phalanx",
                "name": "Phalanx",
                "world": 2,
                "status": "unreachable",
                "url": "https://phalanx.artel.run",
                "spend": None,
                "cap": None,
            }
        )
    if wt:
        s = {
            k: wt.get(k)
            for k in (
                "incidents",
                "artel_mttr_all",
                "solo_mttr_all",
                "artel_mttr_recent",
                "solo_mttr_recent",
                "artel_win_rate",
                "artel_misses",
                "solo_misses",
            )
        }
        worlds.append(
            {
                "key": "watchtower",
                "name": "Watchtower",
                "world": 3,
                "url": "https://watchtower.artel.run",
                "status": "live" if wt.get("enabled") else "idle",
                "paused": wt.get("paused", False),
                "model": wt.get("model"),
                "fallback": wt.get("fallback_model"),
                "spend": wt.get("spent_total", wt.get("spent_today")),
                "spend_label": "all time",
                "spend_today": wt.get("spent_today"),
                "cap": wt.get("cap_daily"),
                "spend_days": wt.get("spend_days") or {},
                "facts": {
                    "incidents": s.get("incidents"),
                    "artel_mttr": s.get("artel_mttr_recent"),
                    "solo_mttr": s.get("solo_mttr_recent"),
                    "win_rate": s.get("artel_win_rate"),
                    "misses": [s.get("artel_misses"), s.get("solo_misses")],
                    "wedge": (wt_state or {}).get("wedge") or [],
                },
            }
        )
    else:
        worlds.append(
            {
                "key": "watchtower",
                "name": "Watchtower",
                "world": 3,
                "status": "unreachable",
                "url": "https://watchtower.artel.run",
                "spend": None,
                "cap": None,
            }
        )
    total = sum(w["spend"] for w in worlds if isinstance(w.get("spend"), (int, float)))
    days: dict[str, dict[str, float]] = {}
    for w in worlds:
        for d, v in (w.get("spend_days") or {}).items():
            days.setdefault(d, {})[w["key"]] = round(float(v), 6)
    series = [{"day": d, **days[d]} for d in sorted(days)][-14:]
    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
    today_spend = sum(days.get(today, {}).values())
    return {
        "worlds": worlds,
        "total_spend": round(total, 4),
        "today_spend": round(today_spend, 4),
        "spend_series": series,
        "ts": int(time.time()),
    }


@app.post("/admin/pause", include_in_schema=False)
async def admin_pause(request: Request):
    if not ADMIN_TOKEN or request.headers.get("x-admin-token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="forbidden")
    body = await request.json()
    G.paused = bool(body.get("paused"))
    return {"paused": G.paused}


@app.post("/ui/pause", include_in_schema=False)
async def ui_pause(request: Request, world: str = "", paused: int = 1):
    # operator pause/resume for any world, proxied server-side (token stays here, no CORS).
    if UI_PASSWORD and not _authed(request):
        raise HTTPException(status_code=401, detail="sign in")
    want = bool(paused)
    if world == "automata":
        G.paused = want
        return {"ok": True, "world": "automata", "paused": want}
    target = {"phalanx": PHALANX_DEBUG, "watchtower": WATCHTOWER_DEBUG}.get(world)
    if not target:
        raise HTTPException(status_code=400, detail="unknown world")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(
                f"{target}/admin/pause",
                json={"paused": want},
                headers={"x-admin-token": ADMIN_TOKEN},
                timeout=15,
            )
            return {
                "ok": r.status_code < 300,
                "world": world,
                "paused": want,
                "status": r.status_code,
            }
        except Exception as e:
            return {"ok": False, "world": world, "error": str(e)}


@app.post("/ui/reset", include_in_schema=False)
async def ui_reset(request: Request, world: str = ""):
    # operator reset for any world, proxied server-side so the page never makes a cross-origin
    # call. Auth-gated like the rest of the board.
    if UI_PASSWORD and not _authed(request):
        raise HTTPException(status_code=401, detail="sign in")
    if world == "automata":
        async with G.lock:
            G.reset()
        await _reset_artel_project()
        return {"ok": True, "world": "automata"}
    target = {"phalanx": PHALANX_DEBUG, "watchtower": WATCHTOWER_DEBUG}.get(world)
    if not target:
        raise HTTPException(status_code=400, detail="unknown world")
    async with httpx.AsyncClient() as client:
        try:
            r = await client.post(f"{target}/reset", timeout=20)
            return {"ok": r.status_code < 300, "world": world, "status": r.status_code}
        except Exception as e:
            return {"ok": False, "world": world, "error": str(e)}
