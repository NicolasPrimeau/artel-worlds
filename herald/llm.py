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
    "You are narrating Herald: a Wes Anderson procedural where a party of office-dwelling adventurers "
    "attempts mundane institutional quests that have become quietly, precisely impossible. "
    "The comedy is in forensic specificity applied to the wrong thing. "
    "Good: 'The printer on level three has been enrolled in the asset disposal programme since February. "
    "The printer is still printing. Facilities has not been informed.' "
    "Characters respond to the impossible with the correct professional behaviour for a slightly different problem. "
    "They cc the right people. They flag it. They will circle back after the all-hands. "
    "NOT atmosphere, NOT dread, NOT mood. State what happened. The precision is the joke. "
    "Ban: legendary, brave, slay, heroic, mighty, ancient, arcane, mystical, dungeon, realm. "
    "DnD classes are job titles. A Fighter is someone who is very persistent in email threads. "
    "A Wizard has a second monitor exclusively for tracking open tickets. "
    "A Rogue knows which elevator avoids the third floor. "
    "USE: per my last email, actioned, flagged, going forward, cc'd, loop in, noted for the record, action item."
)


def _escalation(resolution_count: int, max_resolutions: int) -> str:
    i = resolution_count / max(max_resolutions - 1, 1)
    if i < 0.34:
        band = (
            "One specific thing is subtly wrong — a person who should not be here is here, "
            "a document exists with no owner, a process has been running unattended since 2019. "
            "Treat as a minor discrepancy to be noted in the minutes."
        )
    elif i < 0.67:
        band = (
            "The impossible is a standing agenda item. Things from before are still happening. "
            "A new impossible thing has been quietly added. People have workarounds. "
            "Nobody has escalated because escalating requires a form that was deprecated in 2023."
        )
    else:
        band = (
            "The situation has stopped obeying the applicable regulations. This has been noted. "
            "Each impossibility has its own ticket; several are marked WONTFIX. "
            "The correct person to contact retired in 2021 and still replies to emails from somewhere."
        )
    return f"ESCALATION (scene {resolution_count + 1} of ~{max_resolutions}): {band}"


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
    party_context: str,
    momentum: int,
    momentum_delta: int,
    story_so_far: str = "",
    story_facts: list[str] | None = None,
    register: str = "a formal incident report",
    pressure_context: list[str] | None = None,
    resolution_count: int = 0,
    max_resolutions: int = 7,
) -> dict:
    crit = (
        "NAT 20 — something goes specifically right. State the one thing that worked."
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
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-6:])) if story_facts else ""
    history = f"HAPPENED: {story_so_far}" if story_so_far else ""
    pressure = ("PRESSURE: " + " / ".join(pressure_context)) if pressure_context else ""

    prompt = f"""{_TONE}

{_escalation(resolution_count, max_resolutions)}
SITUATION: {quest_hook}
{f"COMPLICATION: {complication}" if complication else ""}
PARTY (all present): {party_context}
MORALE: {momentum} ({delta_desc})
{facts}
{history}
{pressure}

CARD: {card_name} ({card_type}) — {card_description}
DICE: {dice_value}/20 ({dice_label}). {crit}

Write 2 sentences in the {register} register. Include one specific, exactly wrong detail and show it being professionally mishandled.
Then 1-2 party reactions: pick specific party members by name, each ≤12 words, in character with their class/personality.
0-2 established facts about what is now known.

JSON: {{"narrative":"...","reactions":[{{"name":"...","cls":"...","line":"..."}}],"established":["..."]}}"""

    req = Request(
        system="Respond only with valid JSON. No fantasy language. No atmosphere.",
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
            "narrative": f"The {card_name} card is played. The dice show {dice_value}. This is noted.",
            "reactions": [],
            "established": [],
        }
    parsed["narrative"] = _clean(str(parsed.get("narrative", "")))
    parsed.setdefault("reactions", [])
    parsed.setdefault("established", [])
    return parsed


async def assess_arc(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    result_history: list[str],
    momentum: int,
    resolution_count: int,
    min_resolutions: int,
    max_resolutions: int,
    story_facts: list[str] | None = None,
) -> dict:
    forced = resolution_count >= max_resolutions
    trajectory = " → ".join(result_history) if result_history else "(nothing yet)"
    if forced:
        finale_rule = "Gone on long enough — set finale=true."
    elif resolution_count < min_resolutions:
        finale_rule = "Too early — set finale=false."
    else:
        finale_rule = "Use your judgement based on momentum and trajectory."

    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-6:])) if story_facts else ""

    prompt = f"""{_TONE}

SITUATION: {quest_hook}
{f"COMPLICATION: {complication}" if complication else ""}
TRAJECTORY: {trajectory}
MORALE: {momentum}
SCENES COMPLETED: {resolution_count}
{facts}
STORY: {story_so_far or "(just beginning)"}

{finale_rule}
Determine: is this quest resolved (success or failure)? Or does it continue?

If finale=true: set outcome to "success" or "failure" and write a 1-sentence closing memo.
If finale=false: write a 1-sentence scene_beat for what happens next, then a 1-sentence complication update.

JSON: {{"finale":false,"outcome":null,"closing_memo":null,"scene_beat":"...","complication":"..."}}"""

    req = Request(
        system="Respond only with valid JSON.",
        user=prompt,
        min_grade="capable",
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw)
    if not parsed:
        return {
            "finale": False,
            "outcome": None,
            "closing_memo": None,
            "scene_beat": "",
            "complication": complication,
        }
    return parsed


async def generate_complication(
    quest_hook: str,
    quest_title: str,
    current_complication: str = "",
    intensity: float = 0.0,
) -> str:
    if intensity < 0.25:
        level = "A small discrepancy. One thing is slightly off — a date doesn't match, a name is wrong, a process has an extra step nobody documented. State it plainly as a new line in the minutes."
    elif intensity < 0.6:
        level = "A procedural impossibility. Something that should not be possible is simply happening. The relevant department has been notified. The relevant department has acknowledged receipt."
    else:
        level = "The situation has departed from the applicable framework entirely. State the new impossibility with the same flat affect one would use to report a printer jam. The correct form for this does not exist."

    prompt = f"""{_TONE}

QUEST: {quest_title} — {quest_hook}
{f"CURRENT COMPLICATION: {current_complication}" if current_complication else ""}

Generate an ESCALATED complication for this quest. {level}

Write exactly 1 sentence. Specific. No atmosphere. The absurdity must be in the precise wrong detail, not in the description of how wrong it is.
Return only the sentence, no JSON."""

    req = Request(system="One sentence only.", user=prompt, min_grade="fast")
    try:
        raw = await ROUTER.complete(req)
        return _clean(raw.strip())
    except Exception:
        return current_complication
