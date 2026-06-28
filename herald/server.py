from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import llm
from .engine import (
    CARD_BY_ID,
    MAX_RESOLUTIONS,
    MAX_SCENE_ROUNDS,
    MIN_RESOLUTIONS,
    CardResolution,
    CardType,
    GameState,
    PlayedCard,
    QuestState,
    advance_window,
    apply_card_effects,
    classify_result,
    deal_hand,
    new_game,
    roll_d20,
    sync_register,
)

log = logging.getLogger("herald")
STATIC = Path(__file__).parent / "static"

DEAL_INTERVAL = 22.0
VOTE_TIMEOUT = 20.0
PRESSURE_DURATION = 3
BATCH_WINDOW = 0.4

_rng = random.SystemRandom()
_state: GameState | None = None
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()
_card_signal = asyncio.Event()


def _is_beat(text: object) -> bool:
    return bool(text) and str(text).strip().lower() not in ("none", "null", "n/a", "na", "tbd")


_card_added_sent: set[str] = set()


def _party_context(state: GameState) -> str:
    return " | ".join(f"{m.name} ({m.cls}): {m.personality}" for m in state.party)


def _story_so_far(state: GameState) -> str:
    return " | ".join(state.quest.beats[-8:]) or "(just beginning)"


def _state_snapshot(state: GameState, include_full: bool = True) -> dict:
    return {
        "run_id": state.run_id,
        "phase": state.phase,
        "tick": state.tick,
        "viewers": len(_clients),
        "quest": {
            "title": state.quest.title,
            "hook": state.quest.hook,
            "complication": state.quest.complication,
            "objectives": state.quest.objectives,
            "momentum": state.quest.momentum,
            "pressure_count": len(state.quest.pressure_pool),
            "beats": state.quest.beats[-4:],
            "resolution_count": state.quest.resolution_count,
            "register": state.quest.register,
            "outcome": state.quest.outcome,
        },
        "party": [
            {
                "id": m.id,
                "name": m.name,
                "cls": m.cls,
                "personality": m.personality,
                "hp": m.hp,
                "status": m.status,
                "sprite": m.sprite,
            }
            for m in state.party
        ],
        "window": {
            "opened_at": state.window.opened_at,
            "closes_at": state.window.closes_at,
            "card_count": len(state.window.cards),
            "resolving": state.window.resolving,
        },
        "log": state.log[-30:],
    }


async def _broadcast(msg: dict) -> None:
    dead = set()
    payload = json.dumps(msg)
    for ws in _clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


async def _escalate_complication(state: GameState) -> None:
    from .engine import intensity as _intensity

    run_id = state.run_id
    try:
        new_comp = await asyncio.wait_for(
            llm.generate_complication(
                quest_hook=state.quest.hook,
                quest_title=state.quest.title,
                current_complication=state.quest.complication,
                intensity=_intensity(state.quest.resolution_count),
            ),
            timeout=6.0,
        )
    except Exception:
        return
    if not _is_beat(new_comp) or _state is None or _state.run_id != run_id:
        return
    _state.quest.complication = new_comp
    _state.log_event("complication", new_comp)
    await asyncio.sleep(1.0)
    if _state is not None and _state.run_id == run_id:
        await _broadcast({"type": "scene_beat", "text": new_comp, "who": ""})


async def _resolve_window(state: GameState) -> None:
    played = advance_window(state, _rng)

    state.window.resolving = True
    state.phase = "resolving"
    if played:
        await _broadcast({"type": "window_closing", "card_count": len(played)})
        await asyncio.sleep(0.3)

    result_history: list[str] = []
    had_nat1 = False
    nat1_narrative = ""
    had_nat20 = False

    for played_card in played:
        card_def = CARD_BY_ID.get(played_card.card_id)
        if not card_def:
            continue

        if card_def.type in (CardType.ACCUMULATE, CardType.TWEAK):
            if played_card.id not in _card_added_sent:
                await _broadcast(
                    {
                        "type": "card_added",
                        "card_name": card_def.name,
                        "card_type": card_def.type.value,
                        "card_description": card_def.description,
                    }
                )
            _card_added_sent.discard(played_card.id)
            if card_def.type == CardType.ACCUMULATE:
                state.quest.pressure_pool.append(
                    {
                        "name": card_def.name,
                        "description": card_def.description,
                        "remaining": PRESSURE_DURATION,
                    }
                )
            apply_card_effects(card_def, played_card.result, state.quest)
            continue

        dice_value = played_card.dice
        result = played_card.result
        result_history.append(result.value)

        if dice_value == 1:
            had_nat1 = True
        if dice_value == 20:
            had_nat20 = True

        pressure_context = [p["description"] for p in state.quest.pressure_pool]
        delta_before = state.quest.momentum
        apply_card_effects(card_def, result, state.quest)
        momentum_delta = state.quest.momentum - delta_before

        try:
            narration = await asyncio.wait_for(
                llm.narrate_card(
                    card_name=card_def.name,
                    card_description=card_def.description,
                    card_type=card_def.type.value,
                    dice_value=dice_value,
                    dice_label=result.value,
                    quest_hook=state.quest.hook,
                    complication=state.quest.complication,
                    party_context=_party_context(state),
                    momentum=state.quest.momentum,
                    momentum_delta=momentum_delta,
                    story_so_far=_story_so_far(state),
                    story_facts=state.quest.facts,
                    register=state.quest.register,
                    pressure_context=pressure_context if pressure_context else None,
                    resolution_count=state.quest.resolution_count,
                    max_resolutions=MAX_RESOLUTIONS,
                ),
                timeout=15.0,
            )
        except Exception:
            narration = {
                "narrative": f"{card_def.name} — {dice_value}/20. The situation continues.",
                "reactions": [],
                "established": [],
            }

        narrative = narration.get("narrative", "")
        reactions = narration.get("reactions", [])
        established = narration.get("established", [])

        if _is_beat(narrative):
            state.quest.beats.append(narrative)

        for fact in established[:2]:
            if _is_beat(fact) and fact not in state.quest.facts:
                state.quest.facts.append(fact)

        if dice_value == 1 and reactions:
            nat1_narrative = reactions[0].get("line", "") if reactions else ""

        await _broadcast(
            {
                "type": "card_resolved",
                "card_name": card_def.name,
                "card_type": card_def.type.value,
                "dice": dice_value,
                "result": result.value,
                "narrative": narrative,
                "reactions": reactions,
                "momentum": state.quest.momentum,
                "pressure_count": len(state.quest.pressure_pool),
            }
        )
        await asyncio.sleep(BATCH_WINDOW)

        for p in state.quest.pressure_pool:
            p["remaining"] -= 1
        state.quest.pressure_pool = [p for p in state.quest.pressure_pool if p["remaining"] > 0]

    scene_rounds = len(state.quest.beats) - state.quest.scene_beat_start
    scene_resolved = False

    if had_nat20 or (played and all(r == CardResolution.HIT.value for r in result_history)):
        scene_resolved = True
        state.quest.resolution_count += 1
        sync_register(state)
    elif scene_rounds >= MAX_SCENE_ROUNDS or had_nat1:
        scene_resolved = True
        state.quest.resolution_count += 1
        sync_register(state)

    if had_nat1:
        await _broadcast({"type": "crit_moment", "crit": "nat_1"})
        if nat1_narrative:
            await asyncio.sleep(0.5)
            await _broadcast({"type": "scene_beat", "text": nat1_narrative, "who": ""})

    if scene_resolved:
        state.quest.scene_beat_start = len(state.quest.beats)

    try:
        arc = await asyncio.wait_for(
            llm.assess_arc(
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                story_so_far=_story_so_far(state),
                result_history=result_history,
                momentum=state.quest.momentum,
                resolution_count=state.quest.resolution_count,
                min_resolutions=MIN_RESOLUTIONS,
                max_resolutions=MAX_RESOLUTIONS,
                story_facts=state.quest.facts,
            ),
            timeout=12.0,
        )
    except Exception:
        arc = {}

    if arc.get("finale"):
        outcome = arc.get("outcome") or ("success" if state.quest.momentum >= 0 else "failure")
        await _end_quest(state, outcome, arc.get("closing_memo", ""))
    else:
        beat = arc.get("scene_beat", "")
        comp = arc.get("complication", "")
        if _is_beat(beat):
            state.quest.beats.append(beat)
            await _broadcast({"type": "scene_beat", "text": beat, "who": ""})
        if _is_beat(comp):
            state.quest.complication = comp
        if scene_resolved and not state.quest.outcome:
            asyncio.create_task(_escalate_complication(state))

    state.window.resolving = False
    state.phase = "active"

    import time as _time

    now = _time.time()
    state.window.cards.clear()
    state.window.opened_at = now
    state.window.closes_at = now + DEAL_INTERVAL

    await _broadcast({"type": "state", **_state_snapshot(state, include_full=False)})


async def _end_quest(state: GameState, outcome: str, memo: str = "") -> None:
    state.quest.outcome = outcome
    state.phase = "ended"

    if _is_beat(memo):
        await _broadcast({"type": "scene_beat", "text": memo, "who": ""})

    await _broadcast(
        {
            "type": "quest_ended",
            "outcome": outcome,
            "title": state.quest.title,
            "resolution_count": state.quest.resolution_count,
            "momentum": state.quest.momentum,
        }
    )

    await asyncio.sleep(8.0)
    await _start_new_quest(state)


async def _start_new_quest(state: GameState) -> None:
    import time as _time

    state.quest = _pick_fresh_quest(state)
    state.run_id = str(uuid.uuid4())[:8]
    state.phase = "active"
    state.tick = 0
    state.log.clear()

    now = _time.time()
    state.window.cards.clear()
    state.window.opened_at = now
    state.window.closes_at = now + DEAL_INTERVAL
    state.window.resolving = False

    await _broadcast({"type": "new_quest", **_state_snapshot(state)})


def _pick_fresh_quest(state: GameState) -> QuestState:
    from .engine import _pick_quest

    fresh = _pick_quest(_rng)
    return fresh


async def _game_loop(state: GameState) -> None:
    import time as _time

    while True:
        try:
            now = _time.time()
            if (
                state.phase == "active"
                and now >= state.window.closes_at
                and not state.window.resolving
            ):
                if state.window.cards:
                    await _resolve_window(state)
                else:
                    state.window.opened_at = now
                    state.window.closes_at = now + DEAL_INTERVAL
                    await _broadcast({"type": "window_reset", "closes_at": state.window.closes_at})
                state.tick += 1
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("game loop error: %s", e)
            await asyncio.sleep(1.0)


def create_app() -> FastAPI:
    import time as _time

    global _state

    app = FastAPI(title="Herald")

    @app.on_event("startup")
    async def _startup() -> None:
        global _state
        now = _time.time()
        _state = new_game(_rng, now)
        _state.phase = "active"
        asyncio.create_task(_game_loop(_state))

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        _clients.add(websocket)
        try:
            if _state:
                await websocket.send_text(json.dumps({"type": "state", **_state_snapshot(_state)}))
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect:
            pass
        finally:
            _clients.discard(websocket)

    @app.post("/play")
    async def play_card(body: dict = Body(...)) -> JSONResponse:
        async with _lock:
            if _state is None or _state.phase != "active":
                return JSONResponse({"error": "not active"}, status_code=400)
            card_id = body.get("card_id", "")
            if card_id not in CARD_BY_ID:
                return JSONResponse({"error": "unknown card"}, status_code=400)
            player_id = body.get("player_id", "anon")
            dice = roll_d20(_rng)
            result, _ = classify_result(dice)
            played = PlayedCard(
                id=str(uuid.uuid4())[:8],
                card_id=card_id,
                player_id=player_id,
                dice=dice,
                result=result,
                played_at=__import__("time").time(),
            )
            _state.window.cards.append(played)
            _card_added_sent.add(played.id)
            await _broadcast(
                {
                    "type": "card_played",
                    "card_id": card_id,
                    "card_name": CARD_BY_ID[card_id].name,
                    "card_type": CARD_BY_ID[card_id].type.value,
                    "player_id": player_id,
                    "card_count": len(_state.window.cards),
                }
            )
            return JSONResponse({"ok": True, "dice": dice, "result": result.value})

    @app.get("/hand")
    async def get_hand() -> JSONResponse:
        card_ids = deal_hand(_rng)
        hand = [
            {
                "id": c,
                "name": CARD_BY_ID[c].name,
                "type": CARD_BY_ID[c].type.value,
                "description": CARD_BY_ID[c].description,
            }
            for c in card_ids
            if c in CARD_BY_ID
        ]
        return JSONResponse(hand)

    @app.get("/state")
    async def get_state() -> JSONResponse:
        if _state is None:
            return JSONResponse({})
        return JSONResponse(_state_snapshot(_state))

    @app.get("/spend")
    async def get_spend() -> JSONResponse:
        return JSONResponse({"spend": round(llm.SPEND(), 4)})

    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))

    return app
