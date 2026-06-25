from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import artel, llm
from .engine import (
    CARD_BY_ID,
    MIN_RESOLUTIONS,
    MAX_SCENE_ROUNDS,
    CardResolution,
    CardType,
    DiceResult,
    GameState,
    NPC,
    PlayedCard,
    apply_card_effects,
    apply_disaster,
    advance_window,
    at_station,
    classify_result,
    deal_hand,
    maybe_travel_event,
    new_game,
    roll_d20,
    step_party,
    sync_target,
)

log = logging.getLogger("vibequest")
STATIC = Path(__file__).parent / "static"

_rng = random.SystemRandom()
_state: GameState | None = None
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()
_travel_processed: set[str] = set()


def _character_desc(state: GameState) -> str:
    c = state.character
    return f"{c.name}: {c.personality}"


def _story_so_far(state: GameState) -> str:
    return " | ".join(state.quest.beats[-8:]) or "(just beginning)"


def _scene_name(state: GameState) -> str:
    if state.world and state.world.waypoint_names:
        idx = min(state.target_idx, len(state.world.waypoint_names) - 1)
        return state.world.waypoint_names[idx]
    return f"Location {state.target_idx + 1}"


def _scene_goal(state: GameState) -> str:
    obj = state.quest.objectives
    idx = state.quest.resolution_count
    return obj[idx] if obj and idx < len(obj) else state.quest.hook


def _scene_beats(state: GameState) -> str:
    start = state.quest.scene_beat_start
    recent = state.quest.beats[start:]
    if not recent:
        return "(nothing yet — first round)"
    return "\n".join(f"Round {i + 1}: {b}" for i, b in enumerate(recent))


def _state_snapshot(state: GameState, include_world: bool = True) -> dict:
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
            "npcs": [
                {
                    "id": n.id,
                    "name": n.name,
                    "role": n.role,
                    "personality": n.personality,
                    "sprite": n.sprite,
                    "waypoint_idx": n.waypoint_idx,
                }
                for n in state.quest.npcs
            ],
            "beats": state.quest.beats[-4:],
            "resolution_count": state.quest.resolution_count,
            "register": state.quest.register,
            "outcome": state.quest.outcome,
        },
        "character": {
            "id": state.character.id,
            "name": state.character.name,
            "role": state.character.role,
            "personality": state.character.personality,
            "hp": state.character.hp,
            "status": state.character.status,
            "sprite": state.character.sprite,
        },
        "world": (state.world.to_dict() if state.world else None) if include_world else None,
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
    global _travel_processed
    _travel_processed.clear()
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

        momentum_before = state.quest.momentum
        apply_card_effects(card_def, dice_result, state.quest)
        momentum_delta = state.quest.momentum - momentum_before

        result = {
            "narrative": card_def.description,
            "consequence": "",
            "reactions": [],
            "established": [],
        }
        if llm.enabled():
            current_npc = next(
                (n for n in state.quest.npcs if n.waypoint_idx == state.target_idx),
                None,
            )
            npc_context = (
                f"{current_npc.name} ({current_npc.role}): {current_npc.personality}"
                if current_npc
                else ""
            )
            result = await llm.narrate_card(
                card_name=card_def.name,
                card_description=card_def.description,
                card_type=card_def.type.value,
                dice_value=dice_value,
                dice_label=dice_result.value,
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                party_summary=_character_desc(state),
                momentum=state.quest.momentum,
                momentum_delta=momentum_delta,
                memory_context=memory_ctx,
                story_so_far=_story_so_far(state),
                story_facts=list(state.quest.facts),
                register=state.quest.register,
                npc_context=npc_context,
            )

        new_facts = [f for f in result.get("established", []) if isinstance(f, str)][:2]
        state.quest.facts.extend(new_facts)
        if len(state.quest.facts) > 24:
            state.quest.facts = state.quest.facts[-24:]

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
                "established": new_facts,
            },
        )

        if result.get("consequence"):
            state.quest.beats.append(result["consequence"])

        if artel.enabled():
            await artel.write_memory(
                f"Quest: {state.quest.hook}. Card: {card_def.name}. Dice: {dice_value}. {result.get('narrative', '')}",
                tags=["vibequest", "resolution", card_def.type.value],
            )

        await _broadcast(
            {"type": "card_resolved", "state": _state_snapshot(state, include_world=False)}
        )
        await asyncio.sleep(3.0)

    window_result = classify_result(state.window.resolutions)
    state.quest.result_history.append(window_result)
    had_nat_20 = any(r.dice_result == DiceResult.NAT_20 for r in state.window.resolutions)
    had_nat_1 = any(r.dice_result == DiceResult.NAT_1 for r in state.window.resolutions)
    if window_result == "disaster":
        victim = apply_disaster(state, _rng)
        if victim:
            status_word = "lost" if victim.status == "lost" else "rattled"
            state.log_event(
                "disaster",
                f"{victim.name} is {status_word}.",
                {"victim": victim.id, "status": victim.status},
            )

    arc = {"finale": False, "outcome": None}
    if llm.enabled():
        try:
            arc = await llm.assess_arc(
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                story_so_far=_story_so_far(state),
                result_history=state.quest.result_history,
                momentum=state.quest.momentum,
                resolution_count=state.quest.resolution_count,
                min_resolutions=MIN_RESOLUTIONS,
                register=state.quest.register,
                story_facts=list(state.quest.facts),
            )
        except Exception:
            pass

    if arc.get("finale"):
        state.quest.outcome = arc.get("outcome") or (
            "success" if state.quest.momentum >= 0 else "failure"
        )
        if arc.get("closing_beat"):
            state.quest.beats.append(arc["closing_beat"])
            state.log_event("closing_beat", arc["closing_beat"])
        state.quest.resolution_count += 1
        sync_target(state)
    else:
        # NAT_20 with no NAT_1: breakthrough forces scene to resolve now
        # NAT_1 (regardless of 20): disaster forces scene to continue
        # Both present: chaotic — let the DM decide normally
        nat20_wins = had_nat_20 and not had_nat_1
        nat1_blocks = had_nat_1

        if nat20_wins:
            state.log_event("crit_breakthrough", "NAT 20 — scene resolved by breakthrough")
            state.quest.resolution_count += 1
            state.quest.scene_beat_start = len(state.quest.beats)
            state.quest.scene_rounds = 0
            sync_target(state)
            await _broadcast({"type": "crit_moment", "crit": "nat_20"})
        else:
            if nat1_blocks:
                await _broadcast({"type": "crit_moment", "crit": "nat_1"})

            scene = {"resolved": False, "dm_note": ""}
            if llm.enabled() and not nat1_blocks:
                try:
                    scene = await llm.assess_scene(
                        scene_name=_scene_name(state),
                        scene_goal=_scene_goal(state),
                        scene_beats=_scene_beats(state),
                        rounds=state.quest.scene_rounds,
                        momentum=state.quest.momentum,
                        max_rounds=MAX_SCENE_ROUNDS,
                        story_facts=list(state.quest.facts),
                    )
                except Exception:
                    pass

            force = state.quest.scene_rounds >= MAX_SCENE_ROUNDS
            if not nat1_blocks and (scene.get("resolved") or force):
                if scene.get("dm_note"):
                    state.log_event("scene_resolved", scene["dm_note"])
                state.quest.resolution_count += 1
                state.quest.scene_beat_start = len(state.quest.beats)
                state.quest.scene_rounds = 0
                sync_target(state)
            else:
                state.quest.scene_rounds += 1
                reason = (
                    "NAT 1 — disaster forces another round"
                    if nat1_blocks
                    else scene.get("dm_note", "")
                )
                if reason:
                    state.log_event("scene_continues", reason)

    await _broadcast(
        {"type": "card_resolved", "state": _state_snapshot(state, include_world=False)}
    )
    await asyncio.sleep(2.0)

    state.window.resolving = False
    state.phase = "active"

    if state.quest.outcome:
        await _end_quest(state)
        return


async def _end_quest(state: GameState) -> None:
    state.phase = "complete"
    closing = ""
    if llm.enabled():
        closing = await llm.narrate_quest_end(
            quest_hook=state.quest.hook,
            outcome=state.quest.outcome or "unclear",
            momentum=state.quest.momentum,
            party_summary=_character_desc(state),
        )
    state.log_event("quest_end", closing or f"The quest concludes. Outcome: {state.quest.outcome}.")
    await _broadcast(
        {
            "type": "quest_complete",
            "outcome": state.quest.outcome,
            "closing": closing,
            "state": _state_snapshot(state, include_world=False),
        }
    )
    await asyncio.sleep(12.0)
    await _start_new_game()


async def _travel_card_loop() -> None:
    while True:
        await asyncio.sleep(3.0)
        state = _state
        if state is None or state.phase != "active" or not _clients:
            continue
        if at_station(state):
            continue
        pending = [c for c in state.window.cards if c.id not in _travel_processed]
        if not pending:
            continue
        card = pending[0]
        _travel_processed.add(card.id)
        card_def = CARD_BY_ID.get(card.card_id)
        if not card_def:
            continue
        dice_value, dice_result = roll_d20(_rng)
        apply_card_effects(card_def, dice_result, state.quest)
        is_chaos_hit = card_def.type == CardType.CHAOS and dice_result in (
            DiceResult.HIGH,
            DiceResult.NAT_20,
        )
        event = ""
        if llm.enabled():
            try:
                if is_chaos_hit:
                    event = await llm.narrate_chaos_interrupt(
                        card_name=card_def.name,
                        card_description=card_def.description,
                        quest_hook=state.quest.hook,
                        story_so_far=_story_so_far(state),
                        dice_value=dice_value,
                    )
                else:
                    event = await llm.narrate_travel_card(
                        card_name=card_def.name,
                        card_type=card_def.type.value,
                        quest_hook=state.quest.hook,
                        story_so_far=_story_so_far(state),
                        dice_value=dice_value,
                        dice_label=dice_result.value,
                    )
            except Exception:
                pass
        if not event:
            event = f"{'Something goes wrong.' if dice_value <= 6 else 'Something happens.'} {card_def.name}."
        state.quest.beats.append(event)
        state.log_event(
            "chaos_interrupt" if is_chaos_hit else "travel_card",
            event,
            {"card": card_def.name, "dice": dice_value},
        )
        await _broadcast({"type": "travel_event", "text": event, "chaos": is_chaos_hit})


async def _start_new_game() -> None:
    global _state
    _state = new_game(_rng)
    if llm.enabled():
        waypoint_count = len(_state.world.waypoints) if _state.world else 5
        theme = _state.world.theme if _state.world else "office"
        opening, objectives, npcs_raw = await asyncio.gather(
            llm.narrate_quest_start(
                quest_hook=_state.quest.hook,
                complication=_state.quest.complication,
                party_summary=_character_desc(_state),
            ),
            llm.generate_objectives(
                quest_hook=_state.quest.hook,
                complication=_state.quest.complication,
            ),
            llm.generate_npcs(
                quest_hook=_state.quest.hook,
                complication=_state.quest.complication,
                theme=theme,
                waypoint_count=waypoint_count,
            ),
            return_exceptions=True,
        )
        if isinstance(opening, str) and opening:
            _state.log_event("opening", opening)
        if isinstance(objectives, list):
            _state.quest.objectives = objectives
        if isinstance(npcs_raw, list):
            available = [s for s in range(1, 11) if s != _state.character.sprite]
            npcs = []
            for i, nd in enumerate(npcs_raw):
                if not isinstance(nd, dict):
                    continue
                sprite = available[i % len(available)] if available else (i % 10 + 1)
                wp_max = waypoint_count - 1
                npcs.append(
                    NPC(
                        id=f"npc_{i}",
                        name=nd.get("name", f"Person {i + 1}"),
                        role=nd.get("role", "Unknown"),
                        personality=nd.get("personality", ""),
                        sprite=sprite,
                        waypoint_idx=min(int(nd.get("waypoint_idx", 0)), wp_max),
                        behavior=nd.get("behavior", "stationary"),
                    )
                )
            _state.quest.npcs = npcs
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
        await asyncio.sleep(5.0)
        if _state is None or _state.phase == "resolving" or _state.phase == "complete":
            continue
        if not _clients:
            continue
        sync_target(_state)
        if not at_station(_state):
            continue
        if not _state.window.cards:
            continue
        async with _lock:
            if _state.window.cards and _state.phase == "active" and at_station(_state):
                await _resolve_window(_state)


def create_app() -> FastAPI:
    app = FastAPI(title="VibeQuest")

    @app.on_event("startup")
    async def startup() -> None:
        global _state
        _state = new_game(_rng)
        asyncio.create_task(_window_loop())
        asyncio.create_task(_move_loop())
        asyncio.create_task(_travel_card_loop())

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
                        "resolution_count": _state.quest.resolution_count,
                        "outcome": _state.quest.outcome,
                    }
                    if _state
                    else None
                ),
                "party_size": 1,
                "spend": round(spend.get("usd", 0.0), 5),
                "spend_days": dict(spend.get("days", {})),
                "calls": spend.get("calls", 0),
                "router": llm.ROUTER.metrics() if llm.enabled() else [],
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
