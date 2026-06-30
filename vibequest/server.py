from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections import deque
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import artel, llm
from .engine import (
    CARD_BY_ID,
    MIN_RESOLUTIONS,
    MAX_RESOLUTIONS,
    SCENE_THRESHOLD,
    CardType,
    GameState,
    PlayedCard,
    QuestState,
    apply_fit_effects,
    apply_world_changes,
    advance_window,
    classify_window,
    deal_hand,
    make_quest,
    new_game,
    step_path,
    sync_target,
)
from .world import find_path

log = logging.getLogger("vibequest")
STATIC = Path(__file__).parent / "static"

DEAL_INTERVAL = 10.0
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
            "scene_progress": state.quest.scene_progress,
            "scene_threshold": SCENE_THRESHOLD,
            "surreal": state.quest.surreal,
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
    if not played:
        return

    state.window.resolving = True
    state.phase = "resolving"
    await _broadcast({"type": "window_closing", "card_count": len(played)})
    await asyncio.sleep(0.3)

    # --- gather the played events (effects come AFTER the LLM rates fit) ---
    mom_before = state.quest.momentum
    prog_before = state.quest.scene_progress
    surreal_before = state.quest.surreal
    plays: list[dict] = []
    target_npc_id: str | None = None
    for played_card in played:
        card_def = CARD_BY_ID.get(played_card.card_id)
        if not card_def:
            continue
        plays.append(
            {
                "name": card_def.name,
                "type": card_def.type.value,
                "description": card_def.description,
            }
        )
        if played_card.target_npc_id and not target_npc_id:
            target_npc_id = played_card.target_npc_id

    if not plays:
        state.window.resolving = False
        state.phase = "active"
        return

    # --- the event lands on the agent where it currently is ---
    current_npc = None
    if target_npc_id:
        current_npc = next((n for n in state.quest.npcs if n.id == target_npc_id), None)
    if current_npc is None:
        current_npc = _npc_near_agent(state)
    npc_context = (
        f"{current_npc.name} ({current_npc.role}): {current_npc.personality}" if current_npc else ""
    )

    memory_ctx = ""
    if artel.enabled():
        try:
            names = " ".join(p["name"] for p in plays)
            memory_ctx = await asyncio.wait_for(
                asyncio.shield(
                    asyncio.create_task(artel.search_memory(f"{state.quest.hook} {names}"))
                ),
                timeout=0.6,
            )
        except Exception:
            pass

    # --- the LLM rates how well the events FIT the moment, and narrates accordingly ---
    result: dict = {
        "fit": 50,
        "narrative": "",
        "reactions": [],
        "established": [],
        "world_changes": [],
    }
    if llm.enabled():
        try:
            result = await llm.narrate_event(
                plays=plays,
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                protagonist=_character_desc(state),
                current_step=_scene_goal(state),
                surreal=state.quest.surreal,
                npc_context=npc_context,
                story_so_far=_story_so_far(state),
                story_facts=list(state.quest.facts),
                memory_context=memory_ctx,
                scene_context=_scene_context(state),
            )
        except Exception as exc:
            log.warning("narrate_event failed: %s", exc)

    # fit drives the meters: high fit -> progress, clash -> surreal
    fit = int(result.get("fit", 50))
    apply_fit_effects(state.quest, fit)
    mom_delta = state.quest.momentum - mom_before
    prog_delta = state.quest.scene_progress - prog_before
    surreal_delta = state.quest.surreal - surreal_before
    scene_resolved = state.quest.scene_progress >= SCENE_THRESHOLD

    new_facts = [f for f in result.get("established", []) if isinstance(f, str)][:2]
    state.quest.facts.extend(new_facts)
    if len(state.quest.facts) > 24:
        state.quest.facts = state.quest.facts[-24:]

    world_changes = [c for c in result.get("world_changes", []) if isinstance(c, dict)][:6]
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

    narrative = result.get("narrative", "")
    if _is_beat(result.get("consequence")):
        state.quest.beats.append(result["consequence"])
    state.log_event(
        "resolution",
        narrative,
        {
            "cards": [p["name"] for p in plays],
            "progress": state.quest.scene_progress,
            "momentum": state.quest.momentum,
            "reactions": result.get("reactions", []),
            "established": new_facts,
        },
    )
    if artel.enabled():
        await artel.write_memory(
            f"Quest: {state.quest.hook}. Cards: {', '.join(p['name'] for p in plays)}. {narrative}",
            tags=["vibequest", "resolution"],
        )

    state.quest.result_history.append(classify_window(prog_delta, mom_delta))

    await _broadcast(
        {
            "type": "card_resolved",
            "narrative": narrative,
            "reactions": result.get("reactions", []),
            "npc_name": current_npc.name if current_npc else "",
            "events": [{"name": p["name"], "kind": p["type"]} for p in plays],
            "momentum_delta": mom_delta,
            "progress_delta": prog_delta,
            "fit": fit,
            "surreal_delta": surreal_delta,
            "agent_sprite": state.character.sprite,
            "npc_sprite": current_npc.sprite if current_npc else 0,
            "state": _state_snapshot(state, include_world=False),
        }
    )
    await asyncio.sleep(0.6)

    # --- scene / objective resolution via progress ---
    if scene_resolved:
        state.quest.scene_progress = 0
        state.quest.scene_beat_start = len(state.quest.beats)
        _complete_artel_objective(state, state.quest.resolution_count)
        state.quest.resolution_count += 1
        if llm.enabled():
            try:
                chosen = await asyncio.wait_for(
                    asyncio.shield(asyncio.create_task(_do_pick_next_waypoint(state))),
                    timeout=1.5,
                )
                if chosen is not None:
                    state.quest.next_waypoint_override = chosen
            except Exception:
                pass
        sync_target(state)

    # --- deterministic quest completion ---
    total_scenes = max(len(state.quest.objectives), MIN_RESOLUTIONS)
    if (
        state.quest.resolution_count >= total_scenes
        or state.quest.resolution_count >= MAX_RESOLUTIONS
    ):
        state.quest.outcome = "success" if state.quest.momentum >= 0 else "failure"

    if scene_resolved and not state.quest.outcome:
        asyncio.create_task(_escalate_complication(state))

    state.window.resolving = False
    state.phase = "active"

    if state.window.cards:
        _card_signal.set()

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
    # retired: all cards resolve through _resolve_window (window-level, no dice)
    return


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
        await asyncio.sleep(0.5)
        state = _state
        if state is None or not _clients or state.phase in ("resolving", "complete"):
            continue
        # free-roam: follow the agent's dynamic path
        if step_path(state):
            await _broadcast(_pos_msg(state))


def _card_msg(card) -> dict:
    return {
        "id": card.id,
        "name": card.name,
        "type": card.type.value,
        "description": card.description,
        "flavor": card.flavor,
    }


_DEAL_TYPE_CYCLE = [CardType.ENCOUNTER, CardType.RIVAL, CardType.BOON, CardType.TWIST]


def _next_deal_card():
    global _deal_type_idx
    target_type = _DEAL_TYPE_CYCLE[_deal_type_idx % len(_DEAL_TYPE_CYCLE)]
    _deal_type_idx += 1
    pool = [c for c in CARD_BY_ID.values() if c.type == target_type]
    return _rng.choices(pool, weights=[c.weight for c in pool])[0]


async def _deal_loop() -> None:
    while True:
        await asyncio.sleep(DEAL_INTERVAL)
        if not _clients or _state is None or _state.phase != "active":
            continue
        await _broadcast({"type": "deal_card", "card": _card_msg(_next_deal_card())})


AGENT_INTERVAL = 8.0
_agent_recent: deque[str] = deque(maxlen=3)


def _npc_tile(state: GameState, npc) -> tuple[int, int]:
    wps = state.world.waypoints
    wp = wps[min(npc.waypoint_idx, len(wps) - 1)]
    return wp[0], wp[1]


def _npc_near_agent(state: GameState):
    if not state.quest.npcs or state.world is None:
        return None
    best, best_d = None, 1e9
    for n in state.quest.npcs:
        tx, ty = _npc_tile(state, n)
        d = abs(tx - state.lx) + abs(ty - state.ly)
        if d < best_d:
            best, best_d = n, d
    return best


async def _agent_loop() -> None:
    # the protagonist is an autonomous agent: it walks to people and talks to them
    while True:
        await asyncio.sleep(AGENT_INTERVAL)
        state = _state
        if state is None or not _clients or state.phase != "active" or state.world is None:
            continue
        if state.path:
            continue  # still walking somewhere
        if state.agent_goal:
            npc = next((n for n in state.quest.npcs if n.id == state.agent_goal), None)
            state.agent_goal = ""
            if npc:
                await _agent_converse(state, npc)
                continue
        await _agent_pick_and_go(state)


async def _agent_pick_and_go(state: GameState) -> None:
    npcs = state.quest.npcs
    if not npcs:
        return
    # the LLM (as the agent) decides WHO to go to and WHY, given its goal
    npc = None
    intent = ""
    if llm.enabled():
        roster = [
            {"id": n.id, "name": n.name, "role": n.role} for n in npcs if n.id not in _agent_recent
        ] or [{"id": n.id, "name": n.name, "role": n.role} for n in npcs]
        try:
            decision = await asyncio.wait_for(
                llm.agent_decide(
                    quest_hook=state.quest.hook,
                    complication=state.quest.complication,
                    story_so_far=_story_so_far(state),
                    npcs=roster,
                    story_facts=list(state.quest.facts),
                ),
                timeout=4.0,
            )
            if decision.get("npc_id"):
                npc = next((n for n in npcs if n.id == decision["npc_id"]), None)
                intent = decision.get("intent", "")
        except Exception:
            pass
    if npc is None:
        options = [n for n in npcs if n.id not in _agent_recent] or list(npcs)
        npc = _rng.choice(options)
    if not intent:
        intent = f"{state.character.name} sets off to find {npc.name}, {npc.role}."
    tx, ty = _npc_tile(state, npc)
    path = find_path(state.world, state.lx, state.ly, tx, ty)
    if path and path[0] == [state.lx, state.ly]:
        path = path[1:]
    state.path = path
    state.agent_goal = npc.id
    _agent_recent.append(npc.id)
    state.log_event("agent_move", intent, {"npc": npc.id})
    await _broadcast({"type": "scene_beat", "text": intent, "who": state.character.name})


async def _agent_converse(state: GameState, npc) -> None:
    if not llm.enabled():
        return
    try:
        result = await asyncio.wait_for(
            llm.agent_converse(
                quest_hook=state.quest.hook,
                complication=state.quest.complication,
                story_so_far=_story_so_far(state),
                agent_name=state.character.name,
                npc_name=npc.name,
                npc_role=npc.role,
                npc_personality=npc.personality,
                story_facts=list(state.quest.facts),
                resolution_count=state.quest.resolution_count,
                max_resolutions=MAX_RESOLUTIONS,
            ),
            timeout=8.0,
        )
    except Exception:
        return
    line = result.get("line", "")
    narrative = result.get("narrative", "")
    for f in result.get("established", [])[:1]:
        if isinstance(f, str) and f.strip() and len(state.quest.facts) < 24:
            state.quest.facts.append(f.strip())
    if line:
        await _broadcast({"type": "npc_speak", "npc_name": npc.name, "line": line})
    if narrative:
        state.quest.beats.append(narrative)
        state.log_event("agent_talk", narrative, {"npc": npc.id})
        await _broadcast(
            {
                "type": "scene_event",
                "text": narrative,
                "state": _state_snapshot(state, include_world=False),
            }
        )


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
        asyncio.create_task(_deal_loop())
        asyncio.create_task(_agent_loop())

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
        # accept plays during resolution too — they queue into the next window
        # instead of being rejected (and silently lost after the card animates away)
        if _state is None or _state.phase not in ("active", "resolving"):
            return JSONResponse({"error": "no active game"}, status_code=400)
        card_id = request_data.get("card_id", "")
        player_id = request_data.get("player_id", "anonymous")
        target_npc_id = request_data.get("target_npc_id") or None
        if card_id not in CARD_BY_ID:
            return JSONResponse({"error": "unknown card"}, status_code=400)
        played = PlayedCard(
            id=uuid.uuid4().hex[:12],
            card_id=card_id,
            player_id=player_id,
            target_npc_id=target_npc_id,
        )
        # bare append is race-free in single-threaded asyncio (no await between
        # advance_window's copy+clear); avoids hanging on the long resolve lock.
        _state.window.cards.append(played)
        _card_signal.set()
        await _broadcast(
            {"type": "card_played", "card_id": card_id, "card_count": len(_state.window.cards)}
        )
        # refill: deal a replacement so the hand stays full instead of draining
        await _broadcast({"type": "deal_card", "card": _card_msg(_next_deal_card())})
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
