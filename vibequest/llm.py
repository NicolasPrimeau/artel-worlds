from __future__ import annotations

from . import env
from llmrouter import Request, Router, build_models, parse_json

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
    "cerebras:gpt-oss-120b",
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
    party_summary: str,
    momentum: int,
    memory_context: str,
    story_so_far: str = "",
    register: str = "a deadpan documentary",
) -> dict:
    crit = ""
    if dice_value == 20:
        crit = "This is a NATURAL 20 — a critical success. Something goes spectacularly, memorably right."
    elif dice_value == 1:
        crit = "This is a NATURAL 1 — a critical failure. Something goes genuinely, hilariously wrong with real consequences."
    story_block = f"STORY SO FAR: {story_so_far}" if story_so_far else ""
    prompt = f"""You are the DM for VibeQuest, where a mundane office errand is treated with total epic seriousness, played in the key of {register}.
The party is on a quest. Players just played a card; resolve it.

QUEST: {quest_hook}
COMPLICATION: {complication}
PARTY: {party_summary}
MOMENTUM: {momentum} (negative = going badly, positive = going well)
{story_block}
MEMORY CONTEXT: {memory_context or "No prior context."}

CARD PLAYED: {card_name} ({card_type})
CARD EFFECT: {card_description}
DICE ROLL: {dice_value}/20 ({dice_label}). {crit}

Resolve this card in 2-3 sentences, in the register of {register}. Let the dice steer the outcome. Be specific to this quest; the party takes absurd things completely seriously.
Then write one reaction line per party member (max 15 words each, in character).

Respond as JSON:
{{
  "narrative": "2-3 sentence narration of what happens",
  "consequence": "one sentence on the immediate consequence for the quest",
  "reactions": [{{"name": "...", "role": "...", "line": "..."}}]
}}"""

    req = Request(
        system="You are a deadpan DM narrator. Respond only with valid JSON.", user=prompt
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed or "narrative" not in parsed:
        return {
            "narrative": f"The {card_name} card takes effect. The dice rolled {dice_value}.",
            "consequence": "The quest continues.",
            "reactions": [],
        }
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
) -> dict:
    forced = resolution_count >= min_resolutions * 2 + 2
    trajectory = " → ".join(result_history) if result_history else "(no windows resolved yet)"
    if forced:
        finale_rule = (
            "The quest has gone on long enough — you MUST set finale=true and write a closing beat."
        )
    elif resolution_count < min_resolutions:
        finale_rule = "It is too early to end — set finale=false."
    else:
        finale_rule = """Decide if the story is done. End it ONLY if one of these is true:
- The quest object/goal has been conclusively reached or conclusively lost (not just going well/badly — actually resolved)
- A dramatic climax just landed and continuing would deflate it
- The arc has a clear emotional shape: setup → complication → turning point → resolution

Do NOT end it just because things are going well or badly. Keep going if the story still has forward momentum."""

    prompt = f"""You are the story editor for VibeQuest, played in the key of {register}.
A mundane office quest is being treated with total epic seriousness. You decide when the story ends.

QUEST: {quest_hook}
COMPLICATION: {complication}

WHAT ACTUALLY HAPPENED (narrative consequences, in order):
{story_so_far or "(nothing yet)"}

WINDOW RESULTS TRAJECTORY: {trajectory}
MOMENTUM: {momentum} (-10 = total failure, +10 = glorious success)

{finale_rule}

If finale=true: write one closing beat sentence (what just resolved, in past tense, max 15 words). Set outcome based on whether the quest goal was met.
If finale=false: return only finale=false.

Respond as JSON:
{{
  "finale": false,
  "outcome": "success" or "failure",
  "closing_beat": "one sentence"
}}"""
    req = Request(system="You are a story editor. Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {"finale": forced, "outcome": "success" if momentum >= 0 else "failure"}
    parsed["finale"] = bool(parsed.get("finale")) or forced
    if parsed["finale"] and "outcome" not in parsed:
        parsed["outcome"] = "success" if momentum >= 0 else "failure"
    return parsed


async def narrate_quest_start(quest_hook: str, complication: str, party_summary: str) -> str:
    prompt = f"""You are the narrator for VibeQuest — a deadpan Wes Anderson-style DnD world.
A new quest begins. Write a dramatic 2-sentence opening narration. Treat the mundane situation with complete seriousness.

QUEST: {quest_hook}
COMPLICATION: {complication}
PARTY: {party_summary}

Be deadpan. Do not use em dashes. Keep it under 60 words."""

    req = Request(system="You are a deadpan DM narrator.", user=prompt)
    return await ROUTER.complete(req)


async def narrate_quest_end(
    quest_hook: str, outcome: str, momentum: int, party_summary: str
) -> str:
    prompt = f"""You are the narrator for VibeQuest.
The quest has ended. Write a 2-sentence closing narration. Outcome: {outcome}. Momentum: {momentum}.

QUEST: {quest_hook}
PARTY: {party_summary}

Deadpan tone. Whether success or failure, the party reports it with equal formality."""

    req = Request(system="You are a deadpan DM narrator.", user=prompt)
    return await ROUTER.complete(req)
