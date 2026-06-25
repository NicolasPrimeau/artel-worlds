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
    momentum_delta: int,
    memory_context: str,
    story_so_far: str = "",
    story_facts: list[str] | None = None,
    register: str = "a deadpan documentary",
    npc_context: str = "",
) -> dict:
    if dice_value == 20:
        crit = (
            "NAT 20 — BREAKTHROUGH: Something goes so specifically and absurdly right that it becomes a turning point. "
            "This is the peak of this scene. Make it concrete, make it count, make it the thing that actually moves the needle. "
            "Your 'established' facts should reflect what just changed in the world."
        )
    elif dice_value == 1:
        crit = (
            "NAT 1 — DISASTER: Something goes so specifically and quietly wrong that the situation gets measurably worse. "
            "Not dramatic — just wrong, in a way that will matter. Name what broke, who got caught, what's now in the way. "
            "Your 'established' facts should reflect this new problem."
        )
    else:
        crit = ""

    delta_desc = (
        f"momentum {'improved' if momentum_delta > 0 else 'dropped'} by {abs(momentum_delta)}"
        if momentum_delta != 0
        else "momentum unchanged"
    )
    facts_block = ""
    if story_facts:
        facts_block = "WHAT IS CURRENTLY TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-10:])
    story_block = f"WHAT HAS HAPPENED: {story_so_far}" if story_so_far else ""

    if npc_context:
        reaction_instruction = (
            "Write exactly 2 reactions: first from the PERSON AT THIS LOCATION (use their exact name, speak in their established voice and personality), "
            "then from the player character. Each reaction under 12 words. Workplace register — tired, distracted, focused on the wrong detail."
        )
    else:
        reaction_instruction = (
            "Write 1 reaction from the player character. Under 12 words. Workplace register."
        )

    prompt = f"""{_TONE}

SITUATION: {quest_hook}
COMPLICATION: {complication}
{f"PERSON AT THIS LOCATION: {npc_context}" if npc_context else ""}
PEOPLE: {party_summary}
MORALE: {momentum} ({delta_desc} from this card — let this show in tone, not in explicit statement)
{facts_block}
{story_block}
CONTEXT: {memory_context or "None."}

CARD PLAYED: {card_name} ({card_type})
CARD EFFECT: {card_description}
DICE: {dice_value}/20 ({dice_label}). {crit}

The engine has already applied the mechanical effect. Narrate what caused that outcome — 2-3 sentences, {register} register.
Be specific to named people, objects, and places already established. Do not contradict anything in WHAT IS CURRENTLY TRUE.
If a chaos card: something unexpected really does happen, name it concretely.

{reaction_instruction}

Then 0-2 "established" facts: short present-tense statements about what is now true in the world because of this. Only write facts that actually change something (an object's state, a person's situation, a location's status). Skip if nothing new was established.

JSON only:
{{
  "narrative": "2-3 sentence narration",
  "consequence": "one sentence, the immediate consequence",
  "reactions": [{{"name": "exact person name", "role": "their role", "line": "under 12 words"}}],
  "established": ["fact 1", "fact 2"]
}}"""

    req = Request(system="Respond only with valid JSON. No fantasy language.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed or "narrative" not in parsed:
        return {
            "narrative": f"The {card_name} card is played. The dice show {dice_value}.",
            "consequence": "The situation continues.",
            "reactions": [],
            "established": [],
        }
    if "established" not in parsed:
        parsed["established"] = []
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

    facts_block = ""
    if story_facts:
        facts_block = "\nWHAT IS CURRENTLY TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-10:])

    prompt = f"""{_TONE}

You are deciding when this situation ends.

SITUATION: {quest_hook}
COMPLICATION: {complication}
{facts_block}
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
    card_name: str,
    card_type: str,
    quest_hook: str,
    story_so_far: str,
    dice_value: int = 10,
    dice_label: str = "mid",
) -> str:
    outcome = (
        "something works out slightly better than expected"
        if dice_value >= 14
        else ("something goes mildly wrong" if dice_value <= 6 else "something happens, neutrally")
    )
    prompt = f"""{_TONE}

Someone played a card while in transit between locations. Something happens right now, mid-walk, because of it.

SITUATION: {quest_hook}
WHAT HAS HAPPENED: {story_so_far or "Nothing yet."}
CARD: {card_name} ({card_type})
DICE: {dice_value}/20 ({dice_label}) — {outcome}

One sentence. Under 20 words. Mundane. Nobody acknowledges it's strange."""
    req = Request(system="Respond in one sentence only. No fantasy language.", user=prompt)
    return await ROUTER.complete(req)


async def narrate_chaos_interrupt(
    card_name: str,
    card_description: str,
    quest_hook: str,
    story_so_far: str,
    dice_value: int,
) -> str:
    prompt = f"""{_TONE}

A chaos card just fired during transit with a very high roll. Something unexpected happens RIGHT NOW — mid-walk, no warning.

SITUATION: {quest_hook}
WHAT HAS HAPPENED: {story_so_far or "Nothing yet."}
CARD: {card_name} — {card_description}
DICE: {dice_value}/20 — this was powerful. Something specific and wrong happens.

One sentence. Under 25 words. Delivered completely matter-of-factly, like it's a normal Tuesday."""
    req = Request(system="Respond in one sentence only. No fantasy language.", user=prompt)
    return await ROUTER.complete(req)


async def generate_npcs(
    quest_hook: str,
    complication: str,
    theme: str,
    waypoint_count: int,
) -> list[dict]:
    prompt = f"""{_TONE}

Generate 3 people who live and work in this place.

SITUATION: {quest_hook}
COMPLICATION: {complication}
SETTING TYPE: {theme}
NUMBER OF LOCATIONS: {waypoint_count}

Each person needs a personality: write it in the third person, deadpan, very specific. Like a Wes Anderson character card.
Focus on one precise detail — a habit, an obsession, a fixed routine, a small grievance — delivered completely flatly.
Not quirky-for-quirky's-sake. Just exactly, specifically who this person is.

Examples of the right register:
- "Has worked in this building for nine years. Still does not have a permanent desk. Has stopped asking."
- "Keeps a printed copy of the seating chart from 2019. Refers to it as the correct one."
- "Is always five minutes early. Mentions this when relevant and also when it is not."
- "Once sent an email about the coffee situation. The email was three paragraphs. Nothing changed."
- "Believes the thermostat in room 4B is set incorrectly. Has documented this."

Assign each person to a location (waypoint_idx 0 to {waypoint_count - 1}) they would naturally inhabit.
behavior is "stationary" (stays put) or "wandering" (moves between locations).
Use common, mundane first names (no fantasy names).

Return exactly 3:
JSON only: {{"npcs": [{{"name": "First Last", "role": "Job Title", "personality": "2-3 sentence description", "waypoint_idx": 0, "behavior": "stationary"}}]}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if parsed and isinstance(parsed.get("npcs"), list):
        return [n for n in parsed["npcs"][:4] if isinstance(n, dict)]
    return []


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
        force_rule = "Too early — always set resolved=false."
    elif rounds >= max_rounds:
        force_rule = "This scene has gone on long enough — always set resolved=true."
    else:
        force_rule = """Set resolved=true ONLY if one of these is true:
- The goal was meaningfully addressed: clear progress made, OR decisively failed
- A natural dramatic peak was reached: setup → complication → outcome has a shape
- The scene has a clear closing beat and continuing would repeat ground already covered

Do NOT end the scene just because things are going well or badly in general. End it when something that matters actually HAPPENED here."""

    facts_block = ""
    if story_facts:
        facts_block = "\nWHAT IS CURRENTLY TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-10:])

    prompt = f"""{_TONE}

You are the DM deciding whether this scene is finished.

LOCATION: {scene_name}
GOAL HERE: {scene_goal}
ROUNDS PLAYED HERE: {rounds}
MOMENTUM: {momentum} (-10 = failing badly, +10 = succeeding)
{facts_block}
WHAT HAS HAPPENED AT THIS LOCATION:
{scene_beats or "(Nothing yet — first round.)"}

{force_rule}

JSON only:
{{
  "resolved": false,
  "dm_note": "one sentence: why the scene ends or why it continues"
}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt)
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {"resolved": rounds >= max_rounds, "dm_note": ""}
    parsed["resolved"] = bool(parsed.get("resolved")) or rounds >= max_rounds
    return parsed


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
