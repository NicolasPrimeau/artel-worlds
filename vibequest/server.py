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

from . import artel, llm
from .engine import (
    CARD_BY_ID,
    MIN_RESOLUTIONS,
    MAX_RESOLUTIONS,
    MAX_SCENE_ROUNDS,
    arc_register,
    CardResolution,
    CardType,
    DiceResult,
    GameState,
    PlayedCard,
    QuestState,
    apply_card_effects,
    apply_disaster,
    apply_world_changes,
    advance_window,
    at_station,
    classify_result,
    deal_hand,
    make_quest,
    maybe_travel_event,
    new_game,
    roll_d20,
    step_party,
    sync_target,
)

log = logging.getLogger("vibequest")
STATIC = Path(__file__).parent / "static"

DEAL_INTERVAL = 20.0
VOTE_TIMEOUT = 25.0
PRESSURE_DURATION = 3
BATCH_WINDOW = 0.5

_rng = random.SystemRandom()
_state: GameState | None = None
_clients: set[WebSocket] = set()
_lock = asyncio.Lock()
_card_signal = asyncio.Event()


def _is_beat(text: object) -> bool:
    return bool(text) and str(text).strip().lower() not in ("none", "null", "n/a", "na", "tbd")


_travel_processed: set[str] = set()
_card_added_sent: set[str] = set()
_vote_options: list[QuestState] | None = None
_votes: dict[int, int] = {}
_deal_type_idx: int = 0


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


def _scene_context(state: GameState) -> str:
    if not state.world:
        return ""
    names = state.world.waypoint_names or []
    total = len(state.world.waypoints)
    lo = max(0, state.target_idx - 1)
    hi = min(total, state.target_idx + 2)
    lines = []
    for i in range(lo, hi):
        name = names[i] if i < len(names) else f"Location {i}"
        tag = " ← PLAYER" if i == state.target_idx else ""
        npcs_here = [n for n in state.quest.npcs if n.waypoint_idx == i]
        props_here = [p for p in state.quest.props if p.waypoint_idx == i]
        parts = [f"{n.name} [id:{n.id}] ({n.role}) — {n.personality[:80]}" for n in npcs_here]
        parts += [f"[{p.label}] [id:{p.id}] — {p.description[:60]}" for p in props_here]
        lines.append(f"  {i}. {name}{tag}: {', '.join(parts) if parts else 'empty'}")
    return "\n".join(lines)


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
            "momentum": state.quest.momentum,
            "pressure_count": len(state.quest.pressure_pool),
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
            "props": [
                {
                    "id": p.id,
                    "label": p.label,
                    "description": p.description,
                    "waypoint_idx": p.waypoint_idx,
                }
                for p in state.quest.props
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
    already_handled = set(_travel_processed)
    _travel_processed.clear()
    played = [c for c in advance_window(state, _rng) if c.id not in already_handled]

    state.window.resolving = True
    state.phase = "resolving"
    if played:
        await _broadcast({"type": "window_closing", "card_count": len(played)})
        await asyncio.sleep(0.35)

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
            else:
                state.quest.register = arc_register(
                    state.quest.resolution_count + 2, MAX_RESOLUTIONS
                )
            apply_card_effects(card_def, DiceResult.MID, state.quest)
            resolution = CardResolution(
                card=played_card,
                card_def=card_def,
                dice_value=10,
                dice_result=DiceResult.MID,
                narrative=card_def.description,
                consequence="",
            )
            state.window.resolutions.append(resolution)
            pressure_narrative = card_def.description
            await asyncio.sleep(0.4)
            state.log_event(
                "pressure_added",
                pressure_narrative or card_def.description,
                {"card": card_def.name, "card_type": card_def.type.value},
            )
            if pressure_narrative:
                await _broadcast(
                    {
                        "type": "card_resolved",
                        "narrative": pressure_narrative,
                        "reactions": [],
                        "npc_name": "",
                        "state": _state_snapshot(state, include_world=False),
                    }
                )
            continue

        dice_value, dice_result = roll_d20(_rng)

        artel_task = (
            asyncio.create_task(artel.search_memory(f"{state.quest.hook} {card_def.name}"))
            if artel.enabled()
            else None
        )

        await _broadcast(
            {
                "type": "dice_roll",
                "dice_value": dice_value,
                "dice_result": dice_result.value,
                "card_name": card_def.name,
                "card_type": card_def.type.value,
                "card_description": card_def.description,
            }
        )

        pressure_context = [f"{p['name']}: {p['description']}" for p in state.quest.pressure_pool]
        state.quest.pressure_pool = [
            {**p, "remaining": p["remaining"] - 1} for p in state.quest.pressure_pool
        ]
        state.quest.pressure_pool = [p for p in state.quest.pressure_pool if p["remaining"] > 0]

        momentum_before = state.quest.momentum
        apply_card_effects(card_def, dice_result, state.quest)
        momentum_delta = state.quest.momentum - momentum_before

        current_npc = None
        npc_context = ""
        if llm.enabled():
            if played_card.target_npc_id:
                current_npc = next(
                    (n for n in state.quest.npcs if n.id == played_card.target_npc_id),
                    None,
                )
            else:
                current_npc = next(
                    (n for n in state.quest.npcs if n.waypoint_idx == state.target_idx),
                    None,
                )
            npc_context = (
                f"{current_npc.name} ({current_npc.role}): {current_npc.personality}"
                if current_npc
                else ""
            )

        memory_ctx = ""
        if artel_task:
            try:
                memory_ctx = await asyncio.wait_for(asyncio.shield(artel_task), timeout=0.6)
            except Exception:
                pass

        result = {
            "narrative": card_def.description,
            "consequence": "",
            "reactions": [],
            "established": [],
        }
        llm_task = (
            asyncio.create_task(
                llm.narrate_card(
                    card_name=card_def.name,
                    card_description=card_def.description,
                    card_type=card_def.type.value,
                    dice_value=dice_value,
                    dice_label=dice_result.value,
                    quest_hook=state.quest.hook,
                    complication=state.quest.complication,
                    protagonist=_character_desc(state),
                    momentum=state.quest.momentum,
                    momentum_delta=momentum_delta,
                    memory_context=memory_ctx,
                    story_so_far=_story_so_far(state),
                    story_facts=list(state.quest.facts),
                    register=state.quest.register,
                    npc_context=npc_context,
                    pressure_context=pressure_context,
                    scene_context=_scene_context(state),
                    resolution_count=state.quest.resolution_count,
                    max_resolutions=MAX_RESOLUTIONS,
                )
            )
            if llm.enabled()
            else None
        )

        await asyncio.sleep(0.6)

        if llm_task:
            try:
                result = await llm_task
            except Exception as exc:
                log.warning("narrate_card failed: %s", exc)

        new_facts = [f for f in result.get("established", []) if isinstance(f, str)][:2]
        state.quest.facts.extend(new_facts)
        if len(state.quest.facts) > 24:
            state.quest.facts = state.quest.facts[-24:]

        world_changes = [c for c in result.get("world_changes", []) if isinstance(c, dict)][:6]
        side_effects: list[dict] = []
        if world_changes:
            side_effects = apply_world_changes(state.quest, world_changes, _rng)

        for fx in side_effects:
            action = fx.get("action", "")
            if action == "npc_say":
                npc_id = str(fx.get("npc_id", ""))
                line = str(fx.get("line", ""))
                speaker = next((n for n in state.quest.npcs if n.id == npc_id), None)
                if speaker and line:
                    await _broadcast({"type": "npc_speak", "npc_name": speaker.name, "line": line})
            elif action == "schedule":
                delay = min(float(fx.get("delay", 30)), 120.0)
                event_text = str(fx.get("event", ""))
                follow = [c for c in fx.get("world_changes", []) if isinstance(c, dict)]

                async def _delayed(
                    d: float = delay,
                    t: str = event_text,
                    wc: list = follow,
                    rid: str = state.run_id,
                ) -> None:
                    await asyncio.sleep(d)
                    async with _lock:
                        if _state.run_id != rid:
                            return
                        if wc:
                            apply_world_changes(_state.quest, wc, _rng)
                        if t:
                            _state.log_event("scene_event", t, {})
                    await _broadcast(
                        {
                            "type": "scene_event",
                            "text": t,
                            "state": _state_snapshot(_state, include_world=False),
                        }
                    )

                asyncio.create_task(_delayed())

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

        if _is_beat(result.get("consequence")):
            state.quest.beats.append(result["consequence"])

        if artel.enabled():
            await artel.write_memory(
                f"Quest: {state.quest.hook}. Card: {card_def.name}. Dice: {dice_value}. {result.get('narrative', '')}",
                tags=["vibequest", "resolution", card_def.type.value],
            )

        await _broadcast(
            {
                "type": "card_resolved",
                "narrative": result.get("narrative", ""),
                "reactions": result.get("reactions", []),
                "npc_name": current_npc.name if current_npc else "",
                "state": _state_snapshot(state, include_world=False),
            }
        )
        await asyncio.sleep(0.8)

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

    # Start waypoint pick in parallel with arc assessment (saves ~1s of latency)
    pick_task = asyncio.create_task(_do_pick_next_waypoint(state)) if llm.enabled() else None

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

    scene_resolved = False
    if arc.get("finale"):
        state.quest.outcome = arc.get("outcome") or (
            "success" if state.quest.momentum >= 0 else "failure"
        )
        if _is_beat(arc.get("closing_beat")):
            state.quest.beats.append(arc["closing_beat"])
            state.log_event("closing_beat", arc["closing_beat"])
        _complete_artel_objective(state, state.quest.resolution_count)
        state.quest.resolution_count += 1
        scene_resolved = True
        if pick_task:
            try:
                chosen = await asyncio.wait_for(asyncio.shield(pick_task), timeout=1.0)
                if chosen is not None:
                    state.quest.next_waypoint_override = chosen
            except Exception:
                pass
        sync_target(state)
    else:
        # NAT_20 with no NAT_1: breakthrough forces scene to resolve now
        # NAT_1 (regardless of 20): disaster forces scene to continue
        # Both present: chaotic — let the DM decide normally
        nat20_wins = had_nat_20 and not had_nat_1
        nat1_blocks = had_nat_1

        if nat20_wins:
            state.log_event("crit_breakthrough", "NAT 20 — scene resolved by breakthrough")
            _complete_artel_objective(state, state.quest.resolution_count)
            state.quest.resolution_count += 1
            state.quest.scene_beat_start = len(state.quest.beats)
            state.quest.scene_rounds = 0
            scene_resolved = True
            if pick_task:
                try:
                    chosen = await asyncio.wait_for(asyncio.shield(pick_task), timeout=1.0)
                    if chosen is not None:
                        state.quest.next_waypoint_override = chosen
                except Exception:
                    pass
            sync_target(state)
            await _broadcast({"type": "crit_moment", "crit": "nat_20"})
        else:
            if nat1_blocks:
                await _broadcast({"type": "crit_moment", "crit": "nat_1"})
                nat1_res = next(
                    (r for r in state.window.resolutions if r.dice_result == DiceResult.NAT_1),
                    None,
                )
                if nat1_res and _is_beat(nat1_res.consequence):
                    await asyncio.sleep(0.5)
                    await _broadcast(
                        {"type": "scene_beat", "text": nat1_res.consequence, "who": ""}
                    )

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
                _complete_artel_objective(state, state.quest.resolution_count)
                state.quest.resolution_count += 1
                state.quest.scene_beat_start = len(state.quest.beats)
                state.quest.scene_rounds = 0
                scene_resolved = True
                if pick_task:
                    try:
                        chosen = await asyncio.wait_for(asyncio.shield(pick_task), timeout=1.0)
                        if chosen is not None:
                            state.quest.next_waypoint_override = chosen
                    except Exception:
                        pass
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

    if pick_task and not pick_task.done():
        pick_task.cancel()

    if scene_resolved and not state.quest.outcome:
        asyncio.create_task(_escalate_complication(state))

    await _broadcast(
        {"type": "card_resolved", "state": _state_snapshot(state, include_world=False)}
    )
    await asyncio.sleep(0.5)

    state.window.resolving = False
    state.phase = "active"

    if state.quest.outcome:
        await _end_quest(state)
        return


async def _end_quest(state: GameState) -> None:
    global _vote_options, _votes
    state.phase = "complete"
    if artel.enabled():
        remaining = state.quest.artel_task_ids[state.quest.resolution_count :]
        outcome_str = state.quest.outcome or "unknown"
        for task_id in remaining:
            if outcome_str == "success":
                asyncio.create_task(artel.complete_task(task_id, outcome="Quest succeeded."))
            else:
                asyncio.create_task(
                    artel.fail_task(task_id, outcome=f"Quest ended: {outcome_str}.")
                )
    closing = ""
    if llm.enabled():
        closing = await llm.narrate_quest_end(
            quest_hook=state.quest.hook,
            outcome=state.quest.outcome or "unclear",
            momentum=state.quest.momentum,
            protagonist=_character_desc(state),
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
    await asyncio.sleep(5.0)
    q1, _ = make_quest(_rng)
    q2, _ = make_quest(_rng)
    _vote_options = [q1, q2]
    _votes = {0: 0, 1: 0, 2: 0}
    await _broadcast(
        {
            "type": "vote_start",
            "timeout": int(VOTE_TIMEOUT),
            "options": [
                {"idx": 0, "title": q1.title, "hook": q1.hook},
                {"idx": 1, "title": q2.title, "hook": q2.hook},
                {"idx": 2, "title": "Surprise Me", "hook": "A completely random quest."},
            ],
        }
    )
    await asyncio.sleep(VOTE_TIMEOUT)
    winner_idx = max(_votes, key=lambda k: _votes[k]) if any(_votes.values()) else 2
    chosen = None if winner_idx == 2 else _vote_options[winner_idx]
    _vote_options = None
    _votes = {}
    await _start_new_game(preset_quest=chosen)


async def _travel_card_loop() -> None:
    while True:
        await asyncio.sleep(1.0)
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

        if card_def.type in (CardType.ACCUMULATE, CardType.TWEAK):
            apply_card_effects(card_def, DiceResult.MID, state.quest)
            if card_def.type == CardType.ACCUMULATE:
                state.quest.pressure_pool.append(
                    {
                        "name": card_def.name,
                        "description": card_def.description,
                        "remaining": PRESSURE_DURATION,
                    }
                )
            else:
                state.quest.register = arc_register(
                    state.quest.resolution_count + 2, MAX_RESOLUTIONS
                )
            state.log_event("pressure_added", card_def.description, {"card": card_def.name})
            _card_added_sent.discard(card.id)
            continue

        dice_value, dice_result = roll_d20(_rng)
        await _broadcast(
            {
                "type": "dice_roll",
                "dice_value": dice_value,
                "dice_result": dice_result.value,
                "card_name": card_def.name,
                "card_type": card_def.type.value,
                "card_description": card_def.description,
            }
        )
        if _state is not state:
            continue
        apply_card_effects(card_def, dice_result, state.quest)
        is_chaos_hit = card_def.type == CardType.CHAOS and dice_result in (
            DiceResult.HIGH,
            DiceResult.NAT_20,
        )
        travel_llm_task = None
        if llm.enabled():
            if is_chaos_hit:
                travel_llm_task = asyncio.create_task(
                    llm.narrate_chaos_interrupt(
                        card_name=card_def.name,
                        card_description=card_def.description,
                        quest_hook=state.quest.hook,
                        story_so_far=_story_so_far(state),
                        dice_value=dice_value,
                    )
                )
            else:
                travel_llm_task = asyncio.create_task(
                    llm.narrate_travel_card(
                        card_name=card_def.name,
                        card_type=card_def.type.value,
                        quest_hook=state.quest.hook,
                        story_so_far=_story_so_far(state),
                        dice_value=dice_value,
                        dice_label=dice_result.value,
                    )
                )
        await asyncio.sleep(1.0)
        event = ""
        if travel_llm_task:
            try:
                event = await travel_llm_task
            except Exception:
                pass
        if not _is_beat(event):
            event = f"{card_def.name} is handled in transit."
        if _state is not state:
            continue
        state.quest.beats.append(event)
        state.log_event(
            "chaos_interrupt" if is_chaos_hit else "travel_card",
            event,
            {"card": card_def.name, "dice": dice_value},
        )
        await _broadcast({"type": "travel_event", "text": event, "chaos": is_chaos_hit})


def _complete_artel_objective(state: GameState, idx: int) -> None:
    if not artel.enabled() or idx >= len(state.quest.artel_task_ids):
        return
    task_id = state.quest.artel_task_ids[idx]
    asyncio.create_task(artel.complete_task(task_id, outcome=f"Resolved at scene {idx + 1}."))


async def _escalate_complication(state: GameState) -> None:
    if not llm.enabled():
        return
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
    await asyncio.sleep(1.2)
    if _state is not None and _state.run_id == run_id:
        await _broadcast({"type": "scene_beat", "text": new_comp, "who": ""})


async def _do_pick_next_waypoint(state: GameState) -> int | None:
    if not state.world or not llm.enabled():
        return None
    names = state.world.waypoint_names or []
    max_wp = len(state.world.waypoints) - 1
    current_n = state.quest.resolution_count + 1
    candidates: list[dict] = []
    for offset in (-1, 0, 1):
        idx = current_n + offset
        if 1 <= idx <= max_wp:
            name = names[idx] if idx < len(names) else f"Location {idx}"
            candidates.append({"idx": idx, "name": name})
    if len(candidates) <= 1:
        return None
    cur_name = (
        names[state.target_idx] if state.target_idx < len(names) else f"Location {state.target_idx}"
    )
    try:
        return await asyncio.wait_for(
            llm.pick_next_waypoint(
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                story_so_far=_story_so_far(state),
                current_location=cur_name,
                candidates=candidates,
            ),
            timeout=3.5,
        )
    except Exception:
        return None


async def _start_new_game(preset_quest: QuestState | None = None) -> None:
    global _state
    _state = new_game(_rng, preset_quest=preset_quest)

    complication = ""
    if llm.enabled():
        try:
            complication = await asyncio.wait_for(
                llm.generate_complication(
                    quest_hook=_state.quest.hook,
                    quest_title=_state.quest.title,
                ),
                timeout=5.0,
            )
        except Exception:
            pass
    _state.quest.complication = complication

    opening_text = _state.quest.hook
    _state.log_event("opening", opening_text)
    if complication:
        _state.log_event("complication", complication)

    if artel.enabled():
        asyncio.create_task(
            artel.write_memory(
                f"New VibeQuest begun: {_state.quest.hook} Complication: {complication}",
                tags=["vibequest", "quest-start"],
            )
        )
        for obj in _state.quest.objectives:
            task_id = await artel.create_task(
                title=f"[{_state.quest.title}] {obj}",
                description=_state.quest.hook,
                tags=["vibequest", f"run:{_state.run_id}"],
            )
            if task_id:
                _state.quest.artel_task_ids.append(task_id)
    await _broadcast(
        {"type": "new_quest", "state": _state_snapshot(_state), "opening": opening_text}
    )
    if complication:
        await asyncio.sleep(2.5)
        await _broadcast({"type": "scene_beat", "text": complication, "who": ""})


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
        maybe_travel_event(state, _rng)
        if step_party(state):
            await _broadcast(_pos_msg(state))


def _card_msg(card) -> dict:
    return {
        "id": card.id,
        "name": card.name,
        "type": card.type.value,
        "description": card.description,
        "flavor": card.flavor,
    }


_DEAL_TYPE_CYCLE = [CardType.ACCUMULATE, CardType.ACTION, CardType.CHAOS, CardType.TWEAK]


async def _deal_loop() -> None:
    global _deal_type_idx
    while True:
        await asyncio.sleep(DEAL_INTERVAL)
        if not _clients or _state is None or _state.phase != "active":
            continue
        target_type = _DEAL_TYPE_CYCLE[_deal_type_idx % len(_DEAL_TYPE_CYCLE)]
        _deal_type_idx += 1
        pool = [c for c in CARD_BY_ID.values() if c.type == target_type]
        card = _rng.choices(pool, weights=[c.weight for c in pool])[0]
        await _broadcast({"type": "deal_card", "card": _card_msg(card)})


AMBIENT_INTERVAL = 22.0


async def _ambient_loop() -> None:
    while True:
        await asyncio.sleep(AMBIENT_INTERVAL)
        state = _state
        if state is None or not _clients or state.phase != "active" or not llm.enabled():
            continue
        npcs_here = [n for n in state.quest.npcs if n.waypoint_idx == state.target_idx]
        npc = npcs_here[0] if npcs_here else None
        if not npc:
            continue
        try:
            result = await llm.narrate_ambient(
                quest_hook=state.quest.hook,
                story_so_far=_story_so_far(state),
                scene_name=_scene_name(state),
                npc_name=npc.name,
                npc_role=npc.role,
                npc_personality=npc.personality,
                story_facts=list(state.quest.facts),
            )
            narrative = result.get("narrative", "")
            line = result.get("line", "")
            if line:
                await _broadcast({"type": "npc_speak", "npc_name": npc.name, "line": line})
            if narrative:
                state.log_event("ambient", narrative)
                await _broadcast(
                    {
                        "type": "scene_event",
                        "text": narrative,
                        "state": _state_snapshot(state, include_world=False),
                    }
                )
        except Exception as exc:
            log.warning("ambient event failed: %s", exc)


async def _window_loop() -> None:
    global _state
    while True:
        try:
            await asyncio.wait_for(_card_signal.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        _card_signal.clear()
        if _state is None or _state.phase == "resolving" or _state.phase == "complete":
            continue
        if not _clients:
            continue
        sync_target(_state)
        if not _state.window.cards:
            continue
        await asyncio.sleep(BATCH_WINDOW)
        async with _lock:
            if _state.window.cards and _state.phase == "active":
                try:
                    await _resolve_window(_state)
                except Exception as exc:
                    log.error("_resolve_window crashed: %s", exc)
                    _state.window.resolving = False
                    _state.phase = "active"


def create_app() -> FastAPI:
    app = FastAPI(title="VibeQuest")

    @app.on_event("startup")
    async def startup() -> None:
        global _state
        _state = new_game(_rng)
        asyncio.create_task(_start_new_game())
        asyncio.create_task(_window_loop())
        asyncio.create_task(_move_loop())
        asyncio.create_task(_travel_card_loop())
        asyncio.create_task(_deal_loop())
        asyncio.create_task(_ambient_loop())

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

    @app.post("/vote")
    async def vote(request_data: dict = Body(...)) -> JSONResponse:
        global _votes
        choice = request_data.get("choice")
        if choice not in (0, 1, 2) or _vote_options is None:
            return JSONResponse({"error": "no vote active or invalid choice"}, status_code=400)
        _votes[choice] = _votes.get(choice, 0) + 1
        await _broadcast({"type": "vote_update", "votes": dict(_votes)})
        return JSONResponse({"ok": True, "votes": dict(_votes)})

    @app.post("/play")
    async def play_card(request_data: dict = Body(...)) -> JSONResponse:
        if _state is None or _state.phase != "active":
            return JSONResponse({"error": "no active game"}, status_code=400)
        card_id = request_data.get("card_id", "")
        player_id = request_data.get("player_id", "anonymous")
        target_npc_id = request_data.get("target_npc_id") or None
        if card_id not in CARD_BY_ID:
            return JSONResponse({"error": "unknown card"}, status_code=400)
        card_def = CARD_BY_ID[card_id]
        played = PlayedCard(
            id=uuid.uuid4().hex[:12],
            card_id=card_id,
            player_id=player_id,
            target_npc_id=target_npc_id,
        )
        async with _lock:
            _state.window.cards.append(played)
        _card_signal.set()
        await _broadcast(
            {"type": "card_played", "card_id": card_id, "card_count": len(_state.window.cards)}
        )
        if card_def.type in (CardType.ACCUMULATE, CardType.TWEAK):
            await _broadcast(
                {
                    "type": "card_added",
                    "card_name": card_def.name,
                    "card_type": card_def.type.value,
                    "card_description": card_def.description,
                }
            )
            _card_added_sent.add(played.id)
        return JSONResponse({"ok": True, "card_count": len(_state.window.cards)})

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        _clients.add(ws)
        if _state:
            await ws.send_text(json.dumps({"type": "state", "state": _state_snapshot(_state)}))
            for card in deal_hand(_rng, size=5):
                await ws.send_text(json.dumps({"type": "deal_card", "card": _card_msg(card)}))
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            _clients.discard(ws)

    return app


app = create_app()
