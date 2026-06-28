from __future__ import annotations

import re

from . import env
from llmrouter import Request, Router, build_models, parse_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean(text: str) -> str:
    t = _THINK.sub("", text or "")
    i = t.lower().find("<think>")
    if i != -1:
        t = t[:i]
    return t.strip()


_KEYS = {
    "groq": env("LLM_KEY"),
    "cerebras": env("CEREBRAS_KEY"),
    "sambanova": env("SAMBANOVA_KEY"),
    "gemini": env("LLM2_KEY"),
}

_DEFAULT_POOL = [
    "groq:llama-3.3-70b-versatile",
    "groq:llama-3.1-8b-instant",
    "groq:meta-llama/llama-4-scout-17b-16e-instruct",
    "groq:openai/gpt-oss-120b",
    "groq:openai/gpt-oss-20b",
    "groq:qwen/qwen3-32b",
    "cerebras:gpt-oss-120b",
    "cerebras:zai-glm-4.7",
    "sambanova:Meta-Llama-3.3-70B-Instruct",
    "gemini:gemini-2.5-flash",
    "gemini:gemini-flash-lite-latest",
]
_POOL = [s for s in env("POOL", ",".join(_DEFAULT_POOL)).split(",") if s.strip()]

ROUTER = Router(
    build_models(_POOL, _KEYS),
    concurrency=int(env("CONCURRENCY", "6")),
    cooldown=float(env("COOLDOWN", "8.0")),
    cost_in_per_m=float(env("COST_IN_PER_M", "0.15")),
    cost_out_per_m=float(env("COST_OUT_PER_M", "0.60")),
)

SPEND = ROUTER.spend

_TONE = (
    "Setting: real modern office. It begins ordinary and escalates into wrongness that nobody acknowledges — everyone acts normal. "
    "Register: Wes Anderson deadpan — precise, symmetrical, dryly funny, understated; specific concrete nouns over adjectives; the uncanny reported as administrative fact. "
    "Flat deadpan workplace tone. Never: quest, realm, dungeon, adventurer, arcane, legendary, brave, slay. "
    "Use: flag, loop in, bandwidth, circle back, per my last email, action item."
)


def _escalation(resolution_count: int, max_resolutions: int) -> str:
    i = resolution_count / max(max_resolutions - 1, 1)
    if i < 0.34:
        band = "Reality is intact. Keep this grounded and ordinary; at most the faintest wrongness, the kind anyone would explain away."
    elif i < 0.67:
        band = "Cracks are showing. Small impossibilities happen and nobody remarks on them. Stay calm and procedural."
    else:
        band = "The rules no longer hold. The absurd is routine now; report it flatly, like minutes of a meeting."
    return f"ESCALATION (scene {resolution_count + 1} of ~{max_resolutions}): {band}"


_WORLD_ACTIONS = """\
world_changes actions (0-3, specific real names and labels):
add_prop: {"action":"add_prop","id":"snake_id","label":"≤4 words","description":"exact","waypoint_idx":N}
remove_prop/prop_update: {"action":"remove_prop","id":"id"} / {"action":"prop_update","id":"id","label":"...","description":"..."}
add_npc: {"action":"add_npc","name":"Full Name","role":"Title","personality":"deadpan 1-sentence fact","waypoint_idx":N,"behavior":"stationary"}
move_npc/remove_npc: {"action":"move_npc","npc_id":"id","waypoint_idx":N} / {"action":"remove_npc","npc_id":"id"}
npc_say: {"action":"npc_say","npc_id":"id","line":"≤15 words in their voice"}
schedule: {"action":"schedule","delay":30,"event":"one sentence","world_changes":[...]}"""


def enabled() -> bool:
    return ROUTER.enabled()


async def narrate_card(
    card_name: str,
    card_description: str,
    card_type: str,
    dice_value: int,
    dice_label: str,
    quest_hook: str,
    complication: str,
    protagonist: str,
    momentum: int,
    momentum_delta: int,
    memory_context: str,
    story_so_far: str = "",
    story_facts: list[str] | None = None,
    register: str = "a deadpan documentary",
    npc_context: str = "",
    pressure_context: list[str] | None = None,
    scene_context: str = "",
    resolution_count: int = 0,
    max_resolutions: int = 8,
) -> dict:
    crit = (
        "NAT 20 — something goes specifically right. This is the turning point."
        if dice_value == 20
        else "NAT 1 — something goes specifically wrong. Name what broke."
        if dice_value == 1
        else ""
    )
    delta_desc = (
        f"{'up' if momentum_delta > 0 else 'down'} {abs(momentum_delta)}"
        if momentum_delta != 0
        else "unchanged"
    )
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-8:])) if story_facts else ""
    history = f"HAPPENED: {story_so_far}" if story_so_far else ""
    pressure = ("PRESSURE: " + " / ".join(pressure_context)) if pressure_context else ""
    stage = f"STAGE:\n{scene_context}" if scene_context else ""
    memory = f"MEMORY: {memory_context}" if memory_context else ""
    npc = f"PERSON HERE: {npc_context}" if npc_context else ""

    if npc_context and card_type == "chaos":
        reactions = "2 reactions — NPC first (specific + immediate, ≤12 words), then protagonist (≤12 words)."
    elif npc_context:
        reactions = "2 reactions — NPC in their established voice (≤12 words), then protagonist (≤12 words)."
    else:
        reactions = "1 reaction from protagonist (≤12 words)."

    prompt = f"""{_TONE}

{_escalation(resolution_count, max_resolutions)}
SITUATION: {quest_hook} | COMPLICATION: {complication}
{npc}
PROTAGONIST: {protagonist} | MORALE: {momentum} ({delta_desc})
{facts}
{history}
{pressure}
{stage}
{memory}

CARD: {card_name} ({card_type}) — {card_description}
DICE: {dice_value}/20 ({dice_label}). {crit}

Write 1 narrative sentence ({register} register). {reactions} 0-2 established facts.
{_WORLD_ACTIONS}

JSON: {{"narrative":"...","consequence":"...","reactions":[{{"name":"...","role":"...","line":"..."}}],"established":["..."],"world_changes":[]}}"""

    req = Request(
        system="Respond only with valid JSON. No fantasy language.",
        user=prompt,
        min_grade="capable",
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed or "narrative" not in parsed:
        raw = await ROUTER.complete(req)
        parsed = parse_json(raw)
    if not parsed or "narrative" not in parsed:
        return {
            "narrative": f"The {card_name} card is played. The dice show {dice_value}.",
            "consequence": "The situation continues.",
            "reactions": [],
            "established": [],
            "world_changes": [],
        }
    parsed["narrative"] = _clean(str(parsed.get("narrative", "")))
    if parsed.get("consequence"):
        parsed["consequence"] = _clean(str(parsed["consequence"]))
    parsed.setdefault("established", [])
    parsed.setdefault("world_changes", [])
    return parsed


async def assess_arc(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    result_history: list[str],
    momentum: int,
    resolution_count: int,
    min_resolutions: int,
    register: str = "a deadpan documentary",
    story_facts: list[str] | None = None,
) -> dict:
    forced = resolution_count >= min_resolutions * 2 + 2
    trajectory = " → ".join(result_history) if result_history else "(nothing yet)"
    if forced:
        finale_rule = "Gone on long enough — set finale=true."
    elif resolution_count < min_resolutions:
        finale_rule = "Too early — set finale=false."
    else:
        finale_rule = "End only if: task conclusively done/failed, OR a climax just landed and continuing deflates it. Don't end just because momentum is high or low."

    facts = ("\nTRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-8:])) if story_facts else ""

    prompt = f"""{_TONE}

SITUATION: {quest_hook} | COMPLICATION: {complication}{facts}
HAPPENED: {story_so_far or "(nothing yet)"}
TRAJECTORY: {trajectory} | MORALE: {momentum}

{finale_rule}

If finale=true: one closing sentence (≤15 words, past tense, flat tone). Set outcome.
JSON: {{"finale":false,"outcome":"success or failure","closing_beat":"..."}}"""

    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {"finale": forced, "outcome": "success" if momentum >= 0 else "failure"}
    parsed["finale"] = bool(parsed.get("finale")) or forced
    if parsed["finale"] and "outcome" not in parsed:
        parsed["outcome"] = "success" if momentum >= 0 else "failure"
    return parsed


async def generate_complication(quest_hook: str, quest_title: str) -> str:
    prompt = f"""{_TONE}

This is the opening beat. It must be completely mundane and ordinary — a plausible small office annoyance, nothing strange yet. The wrongness comes later.
One complication sentence, specific to this task, stated flatly.
TASK: {quest_title} — {quest_hook}
One sentence only. No setup, no explanation."""

    req = Request(
        system="Respond with one sentence only. No preamble, no quotation marks.",
        user=prompt,
        min_grade="fast",
        timeout=5.0,
    )
    return _clean(await ROUTER.complete(req))


async def narrate_quest_end(quest_hook: str, outcome: str, momentum: int, protagonist: str) -> str:
    prompt = f"""{_TONE}

Closing: 2 sentences. Outcome: {outcome}. Morale: {momentum}. Flat administrative tone regardless of result.
SITUATION: {quest_hook} | PROTAGONIST: {protagonist}
Under 50 words. No em dashes."""

    req = Request(system="Respond in plain prose. No fantasy language.", user=prompt)
    return _clean(await ROUTER.complete(req))


async def narrate_ambient(
    quest_hook: str,
    story_so_far: str,
    scene_name: str,
    npc_name: str,
    npc_role: str,
    npc_personality: str,
    story_facts: list[str] | None = None,
) -> dict:
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-5:])) if story_facts else ""
    prompt = f"""{_TONE}

The scene ticks forward passively. No card played.
SITUATION: {quest_hook} | LOCATION: {scene_name}
PERSON HERE: {npc_name} ({npc_role}) — {npc_personality}
HAPPENED: {story_so_far or "Nothing yet."}
{facts}

1 observation sentence (≤18 words). 1 line this person says (≤12 words, in their voice, mundane).
JSON: {{"narrative":"...","line":"..."}}"""

    req = Request(system="Respond only with valid JSON. No fantasy language.", user=prompt)
    raw = await ROUTER.complete(req)
    return parse_json(raw) or {}


async def narrate_travel_card(
    card_name: str,
    card_type: str,
    quest_hook: str,
    story_so_far: str,
    dice_value: int = 10,
    dice_label: str = "mid",
) -> str:
    outcome = (
        "slightly better than expected"
        if dice_value >= 14
        else ("mildly wrong" if dice_value <= 6 else "neutral")
    )
    prompt = f"""{_TONE}

A card played mid-transit. Something happens right now, mid-walk.
SITUATION: {quest_hook} | HAPPENED: {story_so_far or "Nothing yet."}
CARD: {card_name} ({card_type}) | DICE: {dice_value}/20 — {outcome}
1 sentence, ≤20 words. Nobody acknowledges it's strange."""

    req = Request(system="Respond in one sentence only. No fantasy language.", user=prompt)
    return _clean(await ROUTER.complete(req))


async def narrate_chaos_interrupt(
    card_name: str,
    card_description: str,
    quest_hook: str,
    story_so_far: str,
    dice_value: int,
) -> str:
    prompt = f"""{_TONE}

Chaos card fired mid-transit. High roll. Something specific and wrong happens right now.
SITUATION: {quest_hook} | HAPPENED: {story_so_far or "Nothing yet."}
CARD: {card_name} — {card_description} | DICE: {dice_value}/20
1 sentence, ≤25 words. Matter-of-fact. Normal Tuesday."""

    req = Request(system="Respond in one sentence only. No fantasy language.", user=prompt)
    return _clean(await ROUTER.complete(req))


async def assess_scene(
    scene_name: str,
    scene_goal: str,
    scene_beats: str,
    rounds: int,
    momentum: int,
    max_rounds: int = 4,
    story_facts: list[str] | None = None,
) -> dict:
    if rounds < 1:
        force_rule = "Too early — resolved=false."
    elif rounds >= max_rounds:
        force_rule = "Gone on long enough — resolved=true."
    else:
        force_rule = "resolved=true only if: goal meaningfully addressed (progress made OR decisively failed), OR a natural dramatic peak was reached. Don't end just because momentum is high or low."

    facts = ("\nTRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-8:])) if story_facts else ""

    prompt = f"""{_TONE}

Decide if this scene is finished.
LOCATION: {scene_name} | GOAL: {scene_goal} | ROUNDS: {rounds} | MORALE: {momentum}{facts}
WHAT HAPPENED HERE:
{scene_beats or "(Nothing yet.)"}

{force_rule}
JSON: {{"resolved":false,"dm_note":"one sentence why"}}"""

    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {"resolved": rounds >= max_rounds, "dm_note": ""}
    parsed["resolved"] = bool(parsed.get("resolved")) or rounds >= max_rounds
    return parsed
