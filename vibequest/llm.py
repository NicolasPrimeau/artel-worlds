from __future__ import annotations

import re

import llmrouter

from . import env
from llmrouter import Request, Router, build_models, parse_json

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _as_text(v) -> str:
    # the LLM sometimes returns a beat as a list of lines — join, don't stringify the list
    if isinstance(v, list):
        return " ".join(_as_text(x) for x in v if str(x).strip())
    return str(v) if v is not None else ""


def _name(v) -> str:
    # speaker labels must be a single clean token — models sometimes cram "Role\nName" in here
    s = " ".join(_as_text(v).split())
    s = s.split(",")[0].split("(")[0].strip()
    return s[:32]


def _clean(text) -> str:
    t = _THINK.sub("", _as_text(text))
    i = t.lower().find("<think>")
    if i != -1:
        t = t[:i]
    t = t.strip()
    # strip stray markdown emphasis the model sometimes leaks (*word*, _word_, **word**)
    t = re.sub(r"\*{1,3}([^*\n]+?)\*{1,3}", r"\1", t)
    t = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"\1", t)
    return t.strip()


_KEYS = {
    "groq": env("LLM_KEY"),
    "cerebras": env("CEREBRAS_KEY"),
    "sambanova": env("SAMBANOVA_KEY"),
    "gemini": env("LLM2_KEY"),
    "openai": env(
        "OPENAI_KEY"
    ),  # set this + add "openai:gpt-4o-mini" to POOL for stronger narration
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
    "openai:gpt-5-nano",
]
_POOL = [s for s in env("POOL", ",".join(_DEFAULT_POOL)).split(",") if s.strip()]

# the router's built-in table doesn't know newer models — they'd default to grade "fast" and get
# skipped for narration. Register them as capable (top grade) + paid so they're eligible.
for _m in ("gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini"):
    llmrouter.CAPS.setdefault(_m, llmrouter.Caps(tier="paid", grade="capable"))

ROUTER = Router(
    build_models(_POOL, _KEYS),
    concurrency=int(env("CONCURRENCY", "6")),
    cooldown=float(env("COOLDOWN", "8.0")),
    cost_in_per_m=float(env("COST_IN_PER_M", "0.15")),
    cost_out_per_m=float(env("COST_OUT_PER_M", "0.60")),
)

# dedicated narration engine: the story beats (resolve_decision, plan_arc) route ONLY here, so they
# consistently use one good model. Falls back to the free pool if the key/model isn't configured.
_NARRATION_SPEC = env("NARRATION_MODEL", "openai:gpt-5-nano")
_narration_models = build_models([_NARRATION_SPEC], _KEYS)
NARRATION = (
    Router(
        _narration_models,
        concurrency=int(env("CONCURRENCY", "6")),
        cooldown=float(env("COOLDOWN", "8.0")),
        cost_in_per_m=float(env("NARR_COST_IN_PER_M", "0.05")),
        cost_out_per_m=float(env("NARR_COST_OUT_PER_M", "0.40")),
    )
    if _narration_models
    else ROUTER
)

SPEND = ROUTER.spend

_TONE = (
    "Narrate VibeQuest in SHORT, punchy beats. ONE to TWO sentences, MAX. Present tense. This is a ticker, NOT a book — never a paragraph. "
    "Natural and a little funny, like a quick workplace-comedy moment. A short line of dialogue is great, but keep each quote ≤10 words and CUT the stage directions ('glanced at his spreadsheet, sighed, and replied' → just the line). "
    "Good: 'Susan asks Jean-Guy to escalate ticket 214. \"Vendor still hasn\\'t called back,\" he shrugs.' "
    "Good: 'The plant is dead. Facilities outsourced watering in March — to nobody.' "
    "Bad (a book): 'Susan leans over Jean-Guy\\'s desk and says, \"Jean-Guy, can we get ticket 214 escalated today so accounting stops breathing down our necks?\" Jean-Guy glances at his spreadsheet, sighs, and replies...' "
    "Dry, deadpan office humour. Specific to THIS moment. No similes, no purple prose, no narrating emotions flatly. "
    "NOT fantasy: it is an office."
)

_TONE_MINI = (
    "You are the DUNGEON MASTER of an office adventure. Narrate like a tabletop DM setting a scene: present "
    "tense, SHORT (1-2 sentences), PLAIN and literal. Describe what is ACTUALLY there — a jammed printer, a "
    "locked supply closet, a manager who won't sign the form — and what it blocks. The tension comes from real "
    "STAKES (what's in the way, who you must get past, the clock), NOT from renaming office things as monsters "
    "or magic. NEVER invent fantasy nouns ('gate', 'sentinel', 'beast', 'ward', 'phantom'); say the real thing "
    "in plain words. A clueless reader must understand exactly what's happening. Deadpan office humour; the "
    "comedy is treating a mundane blocker with a straight, quest-like seriousness. ≤10-word dialogue is fine. "
    "No purple prose. It's a real office played like a quest — the 'spells' are just emails, forms, and coffee."
)


def _escalation(resolution_count: int, max_resolutions: int) -> str:
    # Anchored to the GOAL getting harder — not random office surrealism (which causes drift).
    i = resolution_count / max(max_resolutions - 1, 1)
    if i < 0.34:
        band = "The goal looks routine, but one small bureaucratic snag stands in the way."
    elif i < 0.67:
        band = (
            "The goal is now genuinely hard — process is fighting back, and a workaround is needed."
        )
    else:
        band = "The goal is absurdly entangled in bureaucracy, but it is still the same goal — push it to a conclusion."
    return (
        f"ESCALATION (scene {resolution_count + 1} of ~{max_resolutions}): {band} Stay on the goal."
    )


_WORLD_ACTIONS = """\
world_changes actions (0-3, specific real names and labels):
add_prop: {"action":"add_prop","id":"snake_id","label":"≤4 words","description":"exact","waypoint_idx":N}
remove_prop/prop_update: {"action":"remove_prop","id":"id"} / {"action":"prop_update","id":"id","label":"...","description":"..."}
add_npc: {"action":"add_npc","name":"Full Name (Canadian public servant — French-Canadian, Anglo, or immigrant name)","role":"Civil-service title (Policy Analyst, Program Officer, ATIP Officer, Records, etc.)","personality":"deadpan 1-sentence fact","waypoint_idx":N,"behavior":"stationary"}
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
    register: str = "a warm wildlife documentary",
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

Write 2 sentences in the {register} register. Include one specific, exactly wrong detail (name, date, ticket number, or bureaucratic process) and show it being professionally mishandled by someone nearby. {reactions} 0-2 established facts.
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
    parsed["narrative"] = _clean(parsed.get("narrative", ""))
    if parsed.get("consequence"):
        parsed["consequence"] = _clean(parsed["consequence"])
    parsed.setdefault("established", [])
    parsed.setdefault("world_changes", [])
    return parsed


_EVENT_KIND = {
    "encounter": "a wild interruption crashes in",
    "rival": "someone blocks or challenges the agent",
    "boon": "something helps the agent",
    "twist": "the situation is reframed",
}


def _surreal_band(surreal: int) -> str:
    # stay grounded until it's genuinely off the rails; then build gradually
    if surreal <= 5:
        return "Reality is normal. Things obey cause and effect. Keep it grounded and realistic."
    if surreal <= 8:
        return (
            "Reality has bent slightly. One small impossibility has crept in; nobody remarks on it."
        )
    if surreal <= 11:
        return (
            "Reality is loose now. Causality is negotiable; the impossible is becoming procedure."
        )
    return "Reality has come undone. Events happen out of order, the timeline contradicts itself, everyone carries on regardless."


async def narrate_event(
    plays: list[dict],
    quest_hook: str,
    complication: str,
    protagonist: str,
    current_step: str = "",
    surreal: int = 0,
    mood: str = "",
    npc_context: str = "",
    story_so_far: str = "",
    story_facts: list[str] | None = None,
    memory_context: str = "",
    scene_context: str = "",
) -> dict:
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-8:])) if story_facts else ""
    history = f"WHAT THE AGENT WAS DOING: {story_so_far}" if story_so_far else ""
    stage = f"STAGE:\n{scene_context}" if scene_context else ""
    npc = f"AGENT IS WITH: {npc_context}" if npc_context else ""
    step = f"WHAT THE AGENT NEEDS RIGHT NOW: {current_step}" if current_step else ""
    mood_line = f"AGENT'S MOOD RIGHT NOW: {mood}" if mood else ""

    event_lines = "\n".join(
        f"- {p['name']} ({_EVENT_KIND.get(p['type'], 'an event')}): {p['description']}"
        for p in plays
    )

    reactions = (
        "2 reactions — whoever the agent is with, in their voice (≤10 words), then the AGENT in their own voice + mood (≤10 words)."
        if npc_context
        else "1 reaction from the AGENT, in their own voice + mood (≤10 words)."
    )

    prompt = f"""{_TONE}

THE AGENT'S GOAL: {quest_hook} | COMPLICATION: {complication}
{npc}
THE AGENT: {protagonist}
{mood_line}
{step}
{facts}
{history}
{stage}
REALITY RIGHT NOW: {_surreal_band(surreal)}
The agent is the character the audience roots for. Their reaction must sound like THEM — their personality and current mood — recurring and a little funny.

The audience just threw these EVENTS at the agent:
{event_lines}

First, rate FIT 0-100: how well do these events fit the agent's GOAL and SITUATION (not just the exact current step — judge against the whole goal generously).
- HIGH fit (60-100): the event is plausibly useful to the goal. Be GENEROUS — if a reasonable person could see how it helps, it's HIGH and the agent USES it to make real, visible progress. Most relevant cards belong here.
  If an event is basically THE THING THE GOAL NEEDS, that's near-perfect — let it RESOLVE the situation directly and satisfyingly.
- MID fit (35-59): tangential — only loosely connected. A small nudge with a little weirdness.
- LOW fit (0-34): genuinely unrelated/impossible here. Reality BENDS to absorb it — surreal, deadpan. The mismatch IS the joke.
A relevant card must visibly MATTER — never shrug one off. When in doubt, rate it higher and let it move the story.

COHESION (most important): connect the event to THIS exact moment. Use the specific people, objects, room, and thread already in play (from the facts and what the agent was just doing). The event must land on the agent's CURRENT task — re-cast a generic event in terms of THIS goal (a "courier" arrives with THIS quest's document; a "fire drill" interrupts THIS specific negotiation). Never narrate it as a free-floating random thing. Even a clash happens HERE, to THIS agent, in THIS situation.

Then write a SHORT beat (1-2 sentences, ≤30 words). It must OPEN with the concrete consequence for the goal — what just changed (e.g. "The form is finally signed." / "Glen is sent back to square one." / "The auditor seizes the file."). One quick line of dialogue is fine (≤10 words). No paragraphs, no padding, no similes. {reactions} 0-2 established facts (on a LOW-fit clash, a fact may be rewritten/contradicted). If an event introduces a new person or object, add it via world_changes.
{_WORLD_ACTIONS}

JSON: {{"fit":<0-100 int>,"narrative":"...","reactions":[{{"name":"...","role":"...","line":"..."}}],"established":["..."],"world_changes":[]}}"""

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
        moves = ", ".join(p["name"] for p in plays) or "nothing"
        return {
            "fit": 50,
            "narrative": f"{moves} happens. The agent takes it in stride.",
            "reactions": [],
            "established": [],
            "world_changes": [],
        }
    try:
        parsed["fit"] = max(0, min(100, int(parsed.get("fit", 50))))
    except (TypeError, ValueError):
        parsed["fit"] = 50
    parsed["narrative"] = _clean(parsed.get("narrative", ""))
    parsed.setdefault("established", [])
    parsed.setdefault("world_changes", [])
    rx = []
    for r in parsed.get("reactions", []):
        if isinstance(r, dict) and _as_text(r.get("line", "")).strip():
            rx.append({"name": _name(r.get("name", "")), "line": _clean(r.get("line", ""))})
    parsed["reactions"] = rx
    return parsed


async def resolve_decision(
    situation: str,
    cards: list[dict],
    quest_hook: str,
    protagonist: str,
    roster: list[dict] | None = None,
    mood: str = "",
    surreal: int = 0,
    npc_context: str = "",
    story_so_far: str = "",
    story_facts: list[str] | None = None,
    planned_next: str = "",
) -> dict:
    facts = ("KNOWN: " + " | ".join(story_facts[-4:])) if story_facts else ""
    history = f"STORY SO FAR: {story_so_far}" if story_so_far else ""
    mood_s = f" (mood: {mood})" if mood else ""
    npc = f"WITH: {npc_context}" if npc_context else ""
    if cards:
        card_lines = "; ".join(f"{c['name']}×{c.get('weight', 1)}" for c in cards)
        cards_block = (
            f"THE PARTY CASTS/USES: {card_lines}. The hero performs the heaviest one. "
            "FIT 0-100 = how well that action suits THIS encounter: right move→HIGH (it lands, they clear it); "
            "wrong/absurd move→LOW (they try it anyway, it backfires hilariously, the encounter gets worse/weirder)."
        )
    else:
        cards_block = "The party hesitates — the hero acts on instinct. FIT ~60."
    people = ", ".join(f"{n['id']}={n['name']}" for n in roster) if roster else ""
    prompt = f"""{_TONE_MINI}

QUEST (the hero's goal, never drift): {quest_hook}
HERO: {protagonist}{mood_s}
{npc}
{facts}
{history}
THE OFFICE-DUNGEON RIGHT NOW: {_surreal_band(surreal)}

ENCOUNTER — the hero faces this: "{situation}"
{cards_block}

You're the DM. Narrate this beat of the delve, picking up from the encounter and the action. Give JSON:
- fit (int)
- narrative: REQUIRED, 1-2 short sentences in DM voice — the hero performs the chosen action and the concrete RESULT (clear on its own, never left to the reaction). Plain literal language: say the real office thing and what happens; NEVER invent fantasy nouns (no 'gate', 'sentinel', 'beast', 'ward') — a clueless reader must understand exactly what happened.
- reactions: 0-1 quick ≤10-word quote ({{"name","line"}}).
- breakthrough: true only if the hero clearly CLEARED this encounter (a wrong/weak action → false, the encounter still blocks them).
- established: 0-1 durable fact (or []).
- if breakthrough — next_situation + next_npc_id. PLANNED next encounter: "{planned_next}". Use it (reworded) if they cleared it cleanly; if the action was wrong/chaotic, set derailed=true and DEVIATE into the off-plan consequence instead. One plain literal sentence, ≤14 words — the next obstacle in THE HERO'S OWN way: a PERSON to get past or a thing to force/slip past (something a tactic can act on, never an abstract confirm/verify task), not a repeat, no invented fantasy nouns.
- next_npc_id: an id from [{people}] or "".

JSON: {{"fit":int,"breakthrough":bool,"derailed":bool,"narrative":"...","reactions":[{{"name":"...","line":"..."}}],"next_situation":"...","next_npc_id":"","established":[]}}"""
    req = Request(
        system="Respond only with valid JSON. No fantasy language.",
        user=prompt,
        min_grade="capable",
        allow_paid=True,
        temperature=1.0,  # gpt-5 reasoning models only accept the default temperature
    )
    raw = await NARRATION.complete(req)
    parsed = parse_json(raw) or {}
    # the OUTCOME beat is required — retry once if the model left it empty
    if not _clean(parsed.get("narrative", "")):
        retry = parse_json(await NARRATION.complete(req)) or {}
        if _clean(retry.get("narrative", "")):
            parsed = retry
    try:
        parsed["fit"] = max(0, min(100, int(parsed.get("fit", 60))))
    except (TypeError, ValueError):
        parsed["fit"] = 60
    parsed["breakthrough"] = bool(parsed.get("breakthrough", False))
    parsed["derailed"] = bool(parsed.get("derailed", False))
    parsed["narrative"] = _clean(parsed.get("narrative", ""))
    parsed["next_situation"] = _clean(parsed.get("next_situation", ""))
    parsed.setdefault("next_npc_id", "")
    parsed.setdefault("established", [])
    parsed.setdefault("world_changes", [])
    rx = []
    for r in parsed.get("reactions", []):
        if isinstance(r, dict) and _as_text(r.get("line", "")).strip():
            rx.append({"name": _name(r.get("name", "")), "line": _clean(r.get("line", ""))})
    parsed["reactions"] = rx
    # never leave the audience without an outcome — synthesize a plain one if needed
    if not parsed["narrative"]:
        name = protagonist.split(":")[0].strip() or "The hero"
        tac = max(cards, key=lambda c: c.get("weight", 1))["name"] if cards else "their move"
        parsed["narrative"] = f"{name} tries {tac}. The encounter holds."
    return parsed


async def first_decision(quest_hook: str, complication: str, protagonist: str) -> str:
    prompt = f"""{_TONE_MINI}

QUEST: {quest_hook}
COMPLICATION: {complication}
HERO: {protagonist}

State the FIRST OBSTACLE the hero meets on this delve — a specific thing standing in THE HERO'S OWN way to the quest. The hero must be the one blocked and the one who has to act. Make it something they beat by ACTING on a clear target — usually a PERSON blocking or refusing them (someone to persuade, pressure, outrank, or bribe), sometimes a physical thing to force or slip past. NEVER an abstract "confirm / verify / find out" task with no one to act on. ONE present-tense sentence, ≤14 words, plain literal language a DM would say out loud. No invented fantasy nouns.
JSON: {{"situation":"..."}}"""
    req = Request(system="Respond only with valid JSON.", user=prompt, min_grade="fast")
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw) or {}
    return _clean(parsed.get("situation", "")) or complication or quest_hook


async def plan_arc(quest_hook: str, complication: str, protagonist: str) -> list[str]:
    name = protagonist.split(":")[0]
    prompt = f"""{_TONE_MINI}

QUEST: {quest_hook}
COMPLICATION: {complication}
HERO: {name}

Plan this office DELVE as a 5-ENCOUNTER quest — a real adventure shape, not a flat list of errands. Each beat is a concrete obstacle the hero must get past (a person who won't help, a locked room, a broken thing, a required sign-off), ESCALATING deeper. Distinct encounters, never rephrasings. Shape (escalation only — DO NOT write these labels in the beats):
1. first obstacle barring the way
2. a new blocker, harder
3. a real setback with a cost
4. the make-or-break — the deadline lands / the auditor arrives / it all converges
5. the last barrier before the goal
Every obstacle must stand in THE HERO'S OWN way to the goal — something THEY have to get past — never a bystander's errand.
CRUCIAL: the hero beats each obstacle by ACTING on a clear target — usually a PERSON blocking or refusing them (someone to persuade, pressure, outrank, or bribe), sometimes a physical thing to force, bypass, or slip past. NEVER an abstract "confirm / verify / find out / check" task with no one to act on and nothing to push against.
Each beat: ONE short present-tense sentence, ≤14 words, PLAIN literal language a DM would say out loud. State the real office thing and how it blocks the hero. NO stage labels, NO invented fantasy nouns.
Good: "Karen guards the supply closet you need and won't unlock it without a form."
Good: "Dave must sign your budget, but he's dodging you and hiding in meetings."
Bad (abstract task, nothing to act on): "The calendar shows a 3pm meeting with no organizer; confirm it has a host."
Bad (blocks a bystander, not the hero): "The coffee machine jams, blocking Becky from serving a coffee run."
JSON: {{"beats":["...","...","...","...","..."]}}"""
    req = Request(
        system="Respond only with valid JSON.",
        user=prompt,
        min_grade="capable",
        allow_paid=True,
        temperature=1.0,
    )
    raw = await NARRATION.complete(req)
    parsed = parse_json(raw) or {}
    beats = [_clean(b) for b in parsed.get("beats", []) if str(b).strip()][:6]
    return beats


async def narrate_meltdown(
    quest_hook: str,
    protagonist: str,
    story_so_far: str = "",
    story_facts: list[str] | None = None,
) -> list[str]:
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-6:])) if story_facts else ""
    prompt = f"""{_TONE}

REALITY MELTDOWN. The audience has tipped the world fully surreal. Causality is gone.
THE AGENT: {protagonist}
THE GOAL (now beyond saving): {quest_hook}
{facts}
SO FAR: {story_so_far or "It built to this."}

Write the CLIMAX as 4 escalating beats — reality coming apart in this office, deadpan, each more impossible than the last, building to a final image. Office things become absurd (the photocopier holds a meeting, Tuesday is cancelled, the agent attends their own going-away party). Everyone treats it as normal. The LAST beat is the agent's own reaction, in their voice — they have made peace with it.
Each beat: 1 SHORT line, ≤14 words, Pokemon-terse. No flowery prose.

JSON: {{"beats":["...","...","...","..."]}}"""
    req = Request(
        system="Respond only with valid JSON. No fantasy language.",
        user=prompt,
        min_grade="capable",
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw) or {}
    beats = [_clean(str(b)) for b in parsed.get("beats", []) if str(b).strip()][:4]
    if not beats:
        beats = [
            "The lights go out, then come back wrong.",
            "The photocopier is chairing the meeting now.",
            "Tuesday has been cancelled, retroactively.",
            f"{protagonist.split(':')[0]} decides this is fine. Truly fine.",
        ]
    return beats


async def assess_arc(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    result_history: list[str],
    momentum: int,
    resolution_count: int,
    min_resolutions: int,
    register: str = "a warm wildlife documentary",
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


async def generate_complication(
    quest_hook: str,
    quest_title: str,
    current_complication: str = "",
    intensity: float = 0.0,
) -> str:
    if intensity < 0.25:
        scale = "completely mundane and ordinary — a plausible small office annoyance, nothing strange yet. The wrongness comes later"
    elif intensity < 0.6:
        scale = "weird — something that makes no procedural sense but is treated as routine. Nobody questions it"
    else:
        scale = "impossible — stated flatly, as if it's a normal Tuesday. Specific and concrete"

    prev = (
        f"\nPREVIOUS COMPLICATION: {current_complication}\nThe new one must escalate or differ."
        if current_complication
        else ""
    )
    prompt = f"""{_TONE}{prev}

One complication sentence, specific to this task. Level: {scale}.
TASK: {quest_title} — {quest_hook}
One sentence only. No setup, no explanation."""

    req = Request(
        system="Respond with one sentence only. No preamble, no quotation marks.",
        user=prompt,
        min_grade="fast",
        timeout=5.0,
    )
    return _clean(await ROUTER.complete(req))


async def pick_next_waypoint(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    current_location: str,
    candidates: list[dict],
) -> int | None:
    options_str = "\n".join(f"  {c['idx']}: {c['name']}" for c in candidates)
    prompt = f"""{_TONE}

SITUATION: {quest_hook} | COMPLICATION: {complication}
HAPPENED: {story_so_far or "(just beginning)"}
CURRENT LOCATION: {current_location}

Where should this person go next? Pick the option that would most interestingly escalate or complicate the situation — not always the obvious choice.
OPTIONS:
{options_str}

Reply with only the index number."""

    req = Request(
        system="Respond with a single integer matching one of the option indices. Nothing else.",
        user=prompt,
        min_grade="fast",
        timeout=3.0,
    )
    raw = _clean(await ROUTER.complete(req))
    try:
        return int(raw.strip())
    except Exception:
        return None


async def narrate_quest_end(
    quest_hook: str, outcome: str, momentum: int, protagonist: str, story_so_far: str = ""
) -> str:
    history = f"WHAT HAPPENED: {story_so_far}" if story_so_far else ""
    prompt = f"""{_TONE}

Recap how this quest went in 2 short sentences — what {protagonist.split(":")[0]} was after, and how it actually ended ({outcome}). Dry, deadpan, specific to what happened. Under 45 words. No em dashes.
GOAL: {quest_hook}
{history}"""

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


async def agent_decide(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    npcs: list[dict],
    story_facts: list[str] | None = None,
    agent_name: str = "The agent",
) -> dict:
    # npcs: [{"id","name","role"}] — the agent (in first person) picks who to approach next and why
    roster = "\n".join(f"- [{n['id']}] {n['name']}, {n['role']}" for n in npcs)
    facts = ("KNOWN:\n" + "\n".join(f"- {f}" for f in story_facts[-6:])) if story_facts else ""
    prompt = f"""{_TONE}

You ARE the protagonist. You have ONE goal and you are working it like a person doing errands.
GOAL: {quest_hook} | COMPLICATION: {complication}
SO FAR: {story_so_far or "Just starting."}
{facts}

PEOPLE YOU COULD GO TALK TO:
{roster}

Pick the ONE person most useful to your goal right now. Then write ONE natural sentence, third person, describing {agent_name} heading off to them — like a line of narration, not a justification. Name them. ≤14 words. e.g. "Pam heads for Operations to track down who waters the south plant."
JSON: {{"npc_id":"<exact id from the list>","intent":"<one natural narration sentence>"}}"""
    req = Request(
        system="Respond only with valid JSON. No fantasy language.", user=prompt, min_grade="fast"
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw) or {}
    valid = {n["id"] for n in npcs}
    if parsed.get("npc_id") not in valid:
        return {}
    parsed["intent"] = _clean(parsed.get("intent", ""))
    return parsed


async def agent_converse(
    quest_hook: str,
    complication: str,
    story_so_far: str,
    agent_name: str,
    npc_name: str,
    npc_role: str,
    npc_personality: str,
    story_facts: list[str] | None = None,
    resolution_count: int = 0,
    max_resolutions: int = 8,
    mood: str = "",
) -> dict:
    facts = ("TRUE:\n" + "\n".join(f"- {f}" for f in story_facts[-6:])) if story_facts else ""
    mood_line = f"{agent_name}'s mood: {mood}." if mood else ""
    prompt = f"""{_TONE}

{_escalation(resolution_count, max_resolutions)}
{agent_name} has walked over to {npc_name} specifically to make progress on the goal.
GOAL: {quest_hook} | COMPLICATION: {complication}
SO FAR: {story_so_far or "Just starting."}
{facts}
{mood_line}
{npc_name} ({npc_role}) — {npc_personality}

Write ONE short beat (1-2 sentences, ≤30 words total): {agent_name} asks {npc_name} a quick specific thing about the goal; {npc_name} answers in their voice — helpful or obstructive, ABOUT THE GOAL. Keep {npc_name}'s quote ≤10 words. Don't repeat questions already asked earlier. Tight, no stage-direction padding. The "narrative" is the beat; "line" is just {npc_name}'s short spoken reply.
JSON: {{"narrative":"short beat with the reply","line":"<{npc_name}'s reply, ≤10 words>","established":["<one durable fact about the goal>"]}}"""
    req = Request(
        system="Respond only with valid JSON. No fantasy language.",
        user=prompt,
        min_grade="capable",
    )
    raw = await ROUTER.complete(req)
    parsed = parse_json(raw) or {}
    if parsed.get("narrative"):
        parsed["narrative"] = _clean(parsed["narrative"])
    parsed.setdefault("established", [])
    return parsed


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
