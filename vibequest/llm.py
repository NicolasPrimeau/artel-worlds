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

_TONE = """TONE RULES — read carefully and do not violate them:
The setting is a real, modern, mundane environment (an office, a pub, a school, a grocery store).
The situation keeps escalating — things get increasingly strange and wrong — but no character ever acknowledges this.
Everyone keeps trying to complete the task as if everything is fine.
The humor is entirely in that gap. It is like Mother! or Beau is Afraid: fever-dream wrongness delivered in a completely normal voice.

Language must be casual modern workplace English:
- Use words like: follow up, circle back, escalate, flag, loop in, touch base, per my last email, outside my scope, action item, bandwidth
- Never use: brave, quest, realm, dungeon, adventurer, slay, arcane, mystical, enchanted, champion, valor, ancient, legendary, or any fantasy vocabulary
- Reactions from people sound like real coworkers: tired, slightly passive-aggressive, very focused on the wrong detail, not noticing the big thing happening around them"""


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
    npc_context: str = "",
) -> dict:
    if dice_value == 20:
        crit = "Critical: something goes spectacularly right, in a way that makes total sense in context but shouldn't."
    elif dice_value == 1:
        crit = "Critical: something goes genuinely, quietly wrong — not dramatic, just wrong in a way that will matter."
    else:
        crit = ""
    story_block = f"WHAT HAS HAPPENED SO FAR: {story_so_far}" if story_so_far else ""
    prompt = f"""{_TONE}

SITUATION: {quest_hook}
COMPLICATION: {complication}
{f"PERSON AT THIS LOCATION: {npc_context}" if npc_context else ""}
PEOPLE: {party_summary}
MORALE: {momentum} (negative = things are going badly, positive = going well — let this show in tone, not in explicit statement)
{story_block}
CONTEXT: {memory_context or "None."}

CARD PLAYED: {card_name} ({card_type})
CARD EFFECT: {card_description}
DICE: {dice_value}/20 ({dice_label}). {crit}

Narrate what happens in 2-3 sentences. Narrative register: {register}.
The dice result steers the outcome — a high roll means something works, a low roll means it doesn't, but always in a mundane, slightly wrong way.
Be specific to this situation. Stay grounded. No fantasy language.

Then write one reaction line per person — max 12 words each, sounds like a real coworker, slightly off.

JSON only:
{{
  "narrative": "2-3 sentence narration",
  "consequence": "one sentence, the immediate consequence for the situation",
  "reactions": [{{"name": "...", "role": "...", "line": "..."}}]
}}"""

    req = Request(system="Respond only with valid JSON. No fantasy language.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed or "narrative" not in parsed:
        return {
            "narrative": f"The {card_name} card is played. The dice show {dice_value}.",
            "consequence": "The situation continues.",
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
    trajectory = " → ".join(result_history) if result_history else "(nothing resolved yet)"
    if forced:
        finale_rule = "It has gone on long enough — set finale=true and write a closing beat."
    elif resolution_count < min_resolutions:
        finale_rule = "Too early to end — set finale=false."
    else:
        finale_rule = """End the story ONLY if one of these is true:
- The original task has been conclusively completed or conclusively failed (not just going well — actually done)
- A climax just landed and continuing would deflate it
- The arc has a clear shape: setup → escalation → breaking point → resolution

Do NOT end it just because things are going well or badly. Keep going if it still has forward energy."""

    prompt = f"""{_TONE}

You are deciding when this situation ends.

SITUATION: {quest_hook}
COMPLICATION: {complication}

WHAT ACTUALLY HAPPENED (in order):
{story_so_far or "(nothing yet)"}

TRAJECTORY: {trajectory}
MORALE: {momentum} (-10 = total failure, +10 = success)

{finale_rule}

If finale=true: write one closing beat sentence (past tense, max 15 words, mundane register). Set outcome based on whether the task was actually completed.
If finale=false: return only finale=false.

JSON only:
{{
  "finale": false,
  "outcome": "success" or "failure",
  "closing_beat": "one sentence"
}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {"finale": forced, "outcome": "success" if momentum >= 0 else "failure"}
    parsed["finale"] = bool(parsed.get("finale")) or forced
    if parsed["finale"] and "outcome" not in parsed:
        parsed["outcome"] = "success" if momentum >= 0 else "failure"
    return parsed


async def narrate_quest_start(quest_hook: str, complication: str, party_summary: str) -> str:
    prompt = f"""{_TONE}

A new situation is beginning. Write a 2-sentence opening.
The first sentence establishes the ordinary task. The second introduces the first hint that something is already slightly off — but stated as if it's completely normal.

SITUATION: {quest_hook}
COMPLICATION: {complication}
PEOPLE: {party_summary}

Under 50 words. No em dashes. Modern, flat, casual tone. No fantasy words."""

    req = Request(system="Respond in plain prose. No fantasy language.", user=prompt)
    return await ROUTER.complete(req)


async def narrate_quest_end(
    quest_hook: str, outcome: str, momentum: int, party_summary: str
) -> str:
    prompt = f"""{_TONE}

The situation has ended. Write a 2-sentence closing statement.
Deliver it like a memo or a debrief. Outcome: {outcome}. Morale: {momentum}.
Whether it went well or badly, report it with the same flat administrative tone.

SITUATION: {quest_hook}
PEOPLE: {party_summary}

Under 50 words. No em dashes."""

    req = Request(system="Respond in plain prose. No fantasy language.", user=prompt)
    return await ROUTER.complete(req)


async def narrate_travel_card(
    card_name: str, card_type: str, quest_hook: str, story_so_far: str
) -> str:
    prompt = f"""{_TONE}

Someone played a card while the group was mid-transit — walking between places.
Something happens immediately because of it. Not a big event. Just something.

SITUATION: {quest_hook}
WHAT HAS HAPPENED: {story_so_far or "Nothing yet."}
CARD: {card_name} ({card_type})

One sentence. Something that happens right now, during the walk.
Under 20 words. Mundane. Nobody says it's strange. It just happens."""
    req = Request(system="Respond in one sentence only. No fantasy language.", user=prompt)
    return await ROUTER.complete(req)


async def generate_npcs(
    quest_hook: str,
    complication: str,
    theme: str,
    waypoint_count: int,
) -> list[dict]:
    prompt = f"""{_TONE}

Generate 3 people who live and work in this place. They are real people doing their jobs.
They will mostly be minding their own business. They may be involved in the situation.

SITUATION: {quest_hook}
COMPLICATION: {complication}
SETTING TYPE: {theme}
NUMBER OF LOCATIONS: {waypoint_count}

Assign each person to a location (waypoint_idx 0 to {waypoint_count - 1}) they would naturally inhabit.
behavior is "stationary" (stays put) or "wandering" (moves between locations).

Return exactly 3:
JSON only: {{"npcs": [{{"name": "First Last", "role": "Job Title", "personality": "one phrase", "waypoint_idx": 0, "behavior": "stationary"}}]}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if parsed and isinstance(parsed.get("npcs"), list):
        return [n for n in parsed["npcs"][:4] if isinstance(n, dict)]
    return []


async def generate_objectives(quest_hook: str, complication: str) -> list[str]:
    prompt = f"""{_TONE}

Write exactly 3 objectives for this situation as a to-do list. Each under 8 words.
They should read like items on a real task list — specific, mundane, slightly wrong.
They should escalate: the first is a normal step, the second is where it starts to get complicated, the third is the thing that needs to actually happen to resolve it.

SITUATION: {quest_hook}
COMPLICATION: {complication}

JSON only: {{"objectives": ["...", "...", "..."]}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if parsed and isinstance(parsed.get("objectives"), list):
        return [str(o) for o in parsed["objectives"][:3]]
    return []
