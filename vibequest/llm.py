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
    register: str = "a deadpan documentary",
) -> dict:
    crit = ""
    if dice_value == 20:
        crit = "This is a NATURAL 20 — a critical success. Something goes spectacularly, memorably right."
    elif dice_value == 1:
        crit = "This is a NATURAL 1 — a critical failure. Something goes genuinely, hilariously wrong with real consequences."
    prompt = f"""You are the DM for VibeQuest, where a mundane office errand is treated with total epic seriousness, played in the key of {register}.
The party is on a quest. Players just played a card; resolve it.

QUEST: {quest_hook}
COMPLICATION: {complication}
PARTY: {party_summary}
MOMENTUM: {momentum} (negative = going badly, positive = going well)
MEMORY CONTEXT: {memory_context or "No prior context."}

CARD PLAYED: {card_name} ({card_type})
CARD EFFECT: {card_description}
DICE ROLL: {dice_value}/20 ({dice_label}). {crit}

Resolve this card in 2-3 sentences, in the register of {register}. Let the dice steer the outcome: high rolls go well, low rolls go badly, and a nat 1 or nat 20 should clearly swing the story. Be specific to this quest; the party takes absurd things completely seriously.
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


_RESULT_NUDGE = {
    "breakthrough": "The last scene ended in a CRITICAL SUCCESS (natural 20). Something went spectacularly right; open a real door, reward them, change their fortunes.",
    "triumph": "The last scene went well. They gained ground.",
    "mixed": "The last scene was a muddle. Nothing is clearly better or worse.",
    "chaotic": "The last scene went wildly off the rails (a nat 20 AND a nat 1). Reality is unstable; lean into the absurd.",
    "setback": "The last scene went badly. Add a real complication.",
    "disaster": "The last scene ended in CATASTROPHE (natural 1). Something went genuinely, irreversibly wrong; a path closes, a threat appears, the situation gets materially worse.",
    "uneventful": "Nothing decisive happened last scene. The journey continues.",
}


async def generate_scene(
    quest_hook: str,
    complication: str,
    party_summary: str,
    register: str,
    story_so_far: str,
    prior_result: str,
    momentum: int,
    tension: int,
    scene_number: int,
    max_scenes: int,
) -> dict:
    first = scene_number <= 1
    where = "OPENING SCENE" if first else f"SCENE {scene_number} of at most {max_scenes}"
    cont = (
        "This is the very first situation the party encounters. Open in a way that could ONLY belong to this specific quest."
        if first
        else f"STORY SO FAR: {story_so_far}\n{_RESULT_NUDGE.get(prior_result, '')} The next situation must follow believably and concretely from that."
    )
    finale_rule = f"If the party has plausibly reached the objective (especially after a breakthrough/triumph) OR has clearly failed (after a disaster), set finale=true and make this the climax. You MUST set finale=true if scene_number is {max_scenes}."
    prompt = f"""You are the DM for VibeQuest, where a mundane office errand is treated with total epic seriousness, played in the key of {register}.

QUEST: {quest_hook}
COMPLICATION: {complication}
PARTY: {party_summary}
MOMENTUM: {momentum} (negative = going badly)  TENSION: {tension}
{where}
{cont}

Invent the NEXT situation the party walks into. Rules:
- Play it in the register of {register}; let that genre shape the mood, threat, and language.
- Be SPECIFIC to THIS quest's people, object, and place. Use concrete details, not generic office tropes.
- Avoid the obvious gag (no printers out of toner, no cold coffee, no stapler jokes unless the quest is literally about them).
- Make it a fresh, surprising obstacle. Do not resolve it; just set the scene.
- SHOW DON'T TELL: the "objective" is a short video-game checkbox goal (imperative, max 6 words). The "opening" is ONE line a party member actually says out loud, in character, reacting to what they see.
{finale_rule}

Respond ONLY as JSON:
{{
  "title": "2-4 word scene title",
  "objective": "imperative goal, max 6 words, e.g. 'Slip past the night guard'",
  "opening": "one short spoken line of party dialogue (max 15 words)",
  "speaker": "the role of whoever says it (e.g. Wizard)",
  "finale": false
}}"""
    req = Request(
        system="You are an inventive DM generating the next scene. Respond only with valid JSON.",
        user=prompt,
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed or "objective" not in parsed:
        return {}
    parsed["finale"] = bool(parsed.get("finale")) or scene_number >= max_scenes
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
