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
    result: str,
    momentum: int,
    resolution_count: int,
    min_resolutions: int,
    register: str = "a deadpan documentary",
) -> dict:
    eligible = resolution_count >= min_resolutions
    forced = resolution_count >= min_resolutions * 2 + 2
    finale_instruction = (
        "You MUST set finale=true — the quest has gone on long enough."
        if forced
        else (
            "Set finale=true if the story has reached a natural conclusion: the quest goal is met or clearly lost, the arc feels complete, and a closing line would land cleanly."
            if eligible
            else "Set finale=false — it is too early to end the quest."
        )
    )
    prompt = f"""You are the DM for VibeQuest, played in the key of {register}.

QUEST: {quest_hook}
COMPLICATION: {complication}
STORY SO FAR: {story_so_far or "(just beginning)"}
LAST WINDOW RESULT: {result}
MOMENTUM: {momentum}

{finale_instruction}

Also write a short moment summary (max 8 words, past tense) capturing what just happened — this goes in the quest log.

Respond as JSON:
{{
  "finale": false,
  "outcome": "success" or "failure" (only if finale=true, based on momentum and story),
  "summary": "short moment summary, max 8 words"
}}"""
    req = Request(system="You are a deadpan DM. Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {
            "finale": forced,
            "outcome": "success" if momentum >= 0 else "failure",
            "summary": "The quest continued.",
        }
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
