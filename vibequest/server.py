from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import artel, llm
from .engine import (
    CARD_BY_ID,
    MAX_SCENES,
    CardResolution,
    GameState,
    PlayedCard,
    Scene,
    apply_card_effects,
    advance_window,
    at_station,
    classify_result,
    current_scene,
    deal_hand,
    fallback_scene,
    maybe_travel_event,
    new_game,
    resolve_scene,
    roll_d20,
    scene_conclusion,
    step_party,
    sync_target,
)

log = logging.getLogger("vibequest")
STATIC = Path(__file__).parent / "static"

_rng = random.SystemRandom()
_state: GameState | None = None
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()


def _party_summary(state: GameState) -> str:
    return ", ".join(f"{m.name} the {m.role}" for m in state.party)


def _scene_dict(state: GameState) -> dict | None:
    scenes = state.quest.scenes
    if not scenes:
        return None
    idx = min(state.quest.resolved, len(scenes) - 1)
    s = scenes[idx]
    return {"title": s.title, "description": s.description, "finale": s.finale}


async def _ensure_scene(state: GameState, idx: int) -> Scene | None:
    quest = state.quest
    if idx < len(quest.scenes):
        return quest.scenes[idx]
    prior = quest.scenes[-1] if quest.scenes else None
    story = "; ".join(f"{s.title} ({s.result})" for s in quest.scenes) or "(just beginning)"
    scene_number = idx + 1
    data: dict = {}
    if llm.enabled():
        try:
            data = await llm.generate_scene(
                quest_hook=quest.hook,
                complication=quest.complication,
                party_summary=_party_summary(state),
                story_so_far=story,
                prior_result=prior.result if prior else "",
                momentum=quest.momentum,
                tension=quest.tension,
                scene_number=scene_number,
                max_scenes=MAX_SCENES,
            )
        except Exception:
            data = {}
    if data and data.get("description"):
        scene = Scene(
            title=data.get("title") or f"Scene {scene_number}",
            description=data["description"],
            finale=bool(data.get("finale")),
        )
    else:
        scene = fallback_scene(state, prior, prior.result if prior else "", scene_number, _rng)
    quest.scenes.append(scene)
    return scene


def _state_snapshot(state: GameState) -> dict:
    return {
        "run_id": state.run_id,
        "phase": state.phase,
        "tick": state.tick,
        "viewers": len(_clients),
        "quest": {
            "title": state.quest.title,
            "hook": state.quest.hook,
            "complication": state.quest.complication,
            "scene_number": state.quest.resolved + 1,
            "resolved": state.quest.resolved,
            "max_scenes": MAX_SCENES,
            "scene": _scene_dict(state),
            "momentum": state.quest.momentum,
            "tension": state.quest.tension,
            "outcome": state.quest.outcome,
        },
        "party": [
            {
                "id": m.id,
                "name": m.name,
                "role": m.role,
                "hp": m.hp,
                "status": m.status,
                "sprite": m.sprite,
            }
            for m in state.party
        ],
        "world": state.world.to_dict() if state.world else None,
        "pos": {
            "x": state.lx,
            "y": state.ly,
            "facing": state.facing,
            "target_idx": state.target_idx,
            "rpos": state.rpos,
        },
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


async def _resolve_window(state: GameState) -> None:
    played = advance_window(state, _rng)

    state.window.resolving = True
    state.phase = "resolving"
    if played:
        await _broadcast({"type": "window_closing", "card_count": len(played)})
        await asyncio.sleep(1.5)

    for played_card in played:
        card_def = CARD_BY_ID.get(played_card.card_id)
        if not card_def:
            continue

        dice_value, dice_result = roll_d20(_rng)
        await _broadcast(
            {
                "type": "dice_roll",
                "dice_value": dice_value,
                "dice_result": dice_result.value,
                "card_name": card_def.name,
            }
        )
        await asyncio.sleep(2.0)

        memory_ctx = ""
        if artel.enabled():
            memory_ctx = await artel.search_memory(f"{state.quest.hook} {card_def.name}")

        result = {"narrative": card_def.description, "consequence": "", "reactions": []}
        if llm.enabled():
            scene = current_scene(state)
            complication = state.quest.complication
            if scene:
                complication = (
                    f"{complication} CURRENT SITUATION: {scene.title} — {scene.description}"
                )
            result = await llm.narrate_card(
                card_name=card_def.name,
                card_description=card_def.description,
                card_type=card_def.type.value,
                dice_value=dice_value,
                dice_label=dice_result.value,
                quest_hook=state.quest.hook,
                complication=complication,
                party_summary=_party_summary(state),
                momentum=state.quest.momentum,
                memory_context=memory_ctx,
            )

        apply_card_effects(card_def, dice_result, state.quest)

        resolution = CardResolution(
            card=played_card,
            card_def=card_def,
            dice_value=dice_value,
            dice_result=dice_result,
            narrative=result.get("narrative", ""),
            consequence=result.get("consequence", ""),
        )
        state.window.resolutions.append(resolution)
        state.log_event(
            "resolution",
            result.get("narrative", ""),
            {
                "card": card_def.name,
                "card_type": card_def.type.value,
                "dice": dice_value,
                "dice_result": dice_result.value,
                "consequence": result.get("consequence", ""),
                "reactions": result.get("reactions", []),
            },
        )

        if artel.enabled():
            await artel.write_memory(
                f"Quest: {state.quest.hook}. Card: {card_def.name}. Dice: {dice_value}. {result.get('narrative', '')}",
                tags=["vibequest", "resolution", card_def.type.value],
            )

        await _broadcast({"type": "card_resolved", "state": _state_snapshot(state)})
        await asyncio.sleep(3.0)

    result = classify_result(state.window.resolutions)
    resolved = resolve_scene(state, result, "")
    if resolved:
        conclusion = scene_conclusion(resolved)
        state.log_event("resolution", conclusion, {"card": resolved.title})
        await _broadcast({"type": "card_resolved", "state": _state_snapshot(state)})
        await asyncio.sleep(2.0)

    state.window.resolving = False
    state.phase = "active"

    if state.quest.outcome:
        await _end_quest(state)
        return

    # generate the NEXT scene now, while the party walks toward it
    nxt = await _ensure_scene(state, state.quest.resolved)
    if nxt:
        await _broadcast(
            {
                "type": "scene",
                "number": state.quest.resolved + 1,
                "scene": {"title": nxt.title, "description": nxt.description, "finale": nxt.finale},
                "state": _state_snapshot(state),
            }
        )


async def _end_quest(state: GameState) -> None:
    state.phase = "complete"
    closing = ""
    if llm.enabled():
        closing = await llm.narrate_quest_end(
            quest_hook=state.quest.hook,
            outcome=state.quest.outcome or "unclear",
            momentum=state.quest.momentum,
            party_summary=_party_summary(state),
        )
    state.log_event("quest_end", closing or f"The quest concludes. Outcome: {state.quest.outcome}.")
    await _broadcast(
        {"type": "quest_complete", "outcome": state.quest.outcome, "state": _state_snapshot(state)}
    )
    await asyncio.sleep(12.0)
    await _start_new_game()


async def _start_new_game() -> None:
    global _state
    _state = new_game(_rng)
    opening = ""
    if llm.enabled():
        opening = await llm.narrate_quest_start(
            quest_hook=_state.quest.hook,
            complication=_state.quest.complication,
            party_summary=_party_summary(_state),
        )
    if opening:
        _state.log_event("opening", opening)
    await _ensure_scene(_state, 0)
    if artel.enabled():
        await artel.write_memory(
            f"New VibeQuest begun: {_state.quest.hook} Complication: {_state.quest.complication}",
            tags=["vibequest", "quest-start"],
        )
    await _broadcast({"type": "new_quest", "state": _state_snapshot(_state)})


def _pos_msg(state: GameState) -> dict:
    return {
        "type": "move",
        "pos": {
            "x": state.lx,
            "y": state.ly,
            "facing": state.facing,
            "target_idx": state.target_idx,
            "rpos": state.rpos,
        },
    }


async def _move_loop() -> None:
    while True:
        await asyncio.sleep(0.9)
        state = _state
        if state is None or not _clients or state.phase in ("resolving", "complete"):
            continue
        sync_target(state)
        event = maybe_travel_event(state, _rng)
        if step_party(state):
            await _broadcast(_pos_msg(state))
        if event:
            state.log_event("travel", event)
            await _broadcast({"type": "travel_event", "text": event})


async def _window_loop() -> None:
    global _state
    while True:
        await asyncio.sleep(2.0)
        if _state is None or _state.phase == "resolving" or _state.phase == "complete":
            continue
        if not _clients:
            continue
        sync_target(_state)
        if not at_station(_state):
            continue  # still travelling — situations only resolve at a station
        now = time.time()
        if now >= _state.window.closes_at:
            async with _lock:
                if (
                    now >= _state.window.closes_at
                    and _state.phase == "active"
                    and at_station(_state)
                ):
                    await _resolve_window(_state)


def create_app() -> FastAPI:
    app = FastAPI(title="VibeQuest")

    @app.on_event("startup")
    async def startup() -> None:
        global _state
        _state = new_game(_rng)
        await _ensure_scene(_state, 0)
        asyncio.create_task(_window_loop())
        asyncio.create_task(_move_loop())

    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC / "favicon.ico")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/debug")
    async def debug() -> JSONResponse:
        live = _state is not None and _state.phase == "active" and bool(_clients)
        spend = llm.SPEND if hasattr(llm, "SPEND") else {}
        return JSONResponse(
            {
                "live": live,
                "viewers": len(_clients),
                "phase": _state.phase if _state else "idle",
                "quest": (
                    {
                        "title": _state.quest.title,
                        "hook": _state.quest.hook,
                        "momentum": _state.quest.momentum,
                        "tension": _state.quest.tension,
                        "scene_number": _state.quest.resolved + 1,
                        "scenes_resolved": _state.quest.resolved,
                        "scene": (
                            _state.quest.scenes[
                                min(_state.quest.resolved, len(_state.quest.scenes) - 1)
                            ].title
                            if _state.quest.scenes
                            else None
                        ),
                        "outcome": _state.quest.outcome,
                    }
                    if _state
                    else None
                ),
                "party_size": len(_state.party) if _state else 0,
                "spend": round(spend.get("usd", 0.0), 5),
                "spend_days": dict(spend.get("days", {})),
                "calls": spend.get("calls", 0),
                "llm_enabled": llm.enabled(),
                "artel_enabled": artel.enabled(),
            }
        )

    @app.get("/state")
    async def state_endpoint() -> JSONResponse:
        if _state is None:
            return JSONResponse({"error": "no game"}, status_code=503)
        return JSONResponse(_state_snapshot(_state))

    @app.get("/hand")
    async def hand_endpoint() -> JSONResponse:
        cards = deal_hand(_rng)
        return JSONResponse(
            [
                {
                    "id": c.id,
                    "name": c.name,
                    "type": c.type.value,
                    "description": c.description,
                    "flavor": c.flavor,
                }
                for c in cards
            ]
        )

    @app.post("/play")
    async def play_card(request_data: dict = Body(...)) -> JSONResponse:
        if _state is None or _state.phase != "active":
            return JSONResponse({"error": "no active game"}, status_code=400)
        card_id = request_data.get("card_id", "")
        player_id = request_data.get("player_id", "anonymous")
        if card_id not in CARD_BY_ID:
            return JSONResponse({"error": "unknown card"}, status_code=400)
        played = PlayedCard(id=str(_rng.random()), card_id=card_id, player_id=player_id)
        async with _lock:
            _state.window.cards.append(played)
        await _broadcast(
            {"type": "card_played", "card_id": card_id, "card_count": len(_state.window.cards)}
        )
        return JSONResponse({"ok": True, "card_count": len(_state.window.cards)})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        _clients.add(ws)
        if _state:
            await ws.send_text(json.dumps({"type": "state", "state": _state_snapshot(_state)}))
            hand = deal_hand(_rng)
            await ws.send_text(
                json.dumps(
                    {
                        "type": "hand",
                        "cards": [
                            {
                                "id": c.id,
                                "name": c.name,
                                "type": c.type.value,
                                "description": c.description,
                                "flavor": c.flavor,
                            }
                            for c in hand
                        ],
                    }
                )
            )
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            _clients.discard(ws)

    return app


app = create_app()
