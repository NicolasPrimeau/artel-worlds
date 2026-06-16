from __future__ import annotations

import json
from typing import Protocol

import httpx

from .genome import (
    OPS,
    REGULATORS,
    TARGETS,
    VAR_MAX,
    VARIABLES,
    VERBS,
    Condition,
    Gene,
    Genome,
    dedupe,
)

# An LLM tribe does not act tick-by-tick. It (re)writes its tribe's GENOME — the
# rule list the cheap CA then executes every tick. Swap models by swapping the
# ModelClient (default: Anthropic Haiku). If authoring fails or is unparseable,
# the tribe simply keeps its current DNA. Fog of war holds: the LLM only ever
# sees a summary of its own tribe.

SYSTEM = (
    "You are the collective intelligence of one tribe of organisms in Automata, a survival game "
    "on a toric hex grid. You do NOT act tick by tick. Instead you PROGRAM your tribe's DNA: an "
    "ordered list of rules that every organism runs each tick (the first rule whose condition "
    "holds fires). Rewrite the DNA to help your tribe survive and spread.\n"
    "A rule is: IF a condition holds, DO an action. A condition is a variable, an operator "
    "(> or <), and a threshold. An action is a verb plus, for divide/migrate, a target.\n"
    "Variables (with ranges): " + ", ".join(f"{v} 0..{VAR_MAX[v]}" for v in VARIABLES) + ".\n"
    "Verbs: metabolize = eat nutrient in your cell for energy, but it emits toxin into that cell "
    "(staying put poisons you); divide = spawn a child into a target neighbor (needs energy >= 10, "
    "splits your energy); migrate = move to a target neighbor; dormant = rest, no toxin.\n"
    "Targets (divide/migrate): " + ", ".join(TARGETS) + ".\n"
    "Death: toxin in your cell >= 50, or energy <= 0, or old age. So write rules that flee toxin, "
    "seek nutrient, and divide when energy is high — most urgent rule first.\n"
    'Reply with ONLY JSON, no prose. Start with "note": one short line (<= 12 words) in your '
    "tribe's own voice — what you're changing and WHY, your selfish read of the moment. Then the "
    'genome: {"note": "...", "regulators": {'
    + ", ".join(f'"{r}": <0-100>' for r in REGULATORS)
    + '}, "behaviors": [{"cond1": {"variable": "...", "op": ">", "threshold": 0}, '
    '"cond2": null, "verb": "...", "target": "..."}, ...]}. At most 8 rules, all distinct.'
)


# Distinct temperaments keep LLM tribes from all converging on the same optimal
# genome. Each LLM tribe is handed one — so the world grows divergent strategies
# (cultures) you can watch compete, instead of identical clones.
PERSONAS = (
    "an aggressive expansionist — divide as fast and often as possible to spread your "
    "lineage, tolerating danger. Growth over safety.",
    "a cautious survivalist — avoid toxin above all and prize longevity. Grow slowly and "
    "steadily; never gamble the whole tribe.",
    "a nomadic forager — keep moving toward the richest nutrient, rarely sitting still; "
    "divide opportunistically while roaming.",
    "a toxin-tolerant pioneer — push into the high-toxin frontier timid tribes flee, and "
    "claim the empty space there.",
    "a patient hoarder — build deep energy reserves by feeding in rich cells, and divide "
    "only when extremely strong.",
    "a swarm — stay tightly clustered and divide into open neighbors to dominate territory "
    "by sheer numbers.",
)


# cumulative prompt-cache instrumentation: cached input vs total input across genome authoring
CACHE = {"cached_in": 0, "input": 0}


class ModelClient(Protocol):
    async def complete(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> str: ...


class ClaudeSDKClient:
    # Claude Agent SDK on the subscription OAuth token (CLAUDE_CODE_OAUTH_TOKEN read by
    # the CLI from the environment) — genome authoring draws the plan's monthly credit
    # and PAUSES at exhaustion; an authoring failure just keeps the tribe's current DNA.
    # Spend is metered at standard API rates from the SDK's own per-call figure.
    def __init__(self, model: str):
        self.model = model
        self.spent = 0.0

    async def complete(
        self, system: str, user: str, max_tokens: int = 900, temperature: float = 1.0
    ) -> str:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        opts = ClaudeAgentOptions(
            model=self.model,
            system_prompt=system,
            max_turns=1,
            allowed_tools=[],
            tools=[],
        )
        result = None
        async for msg in query(prompt=user, options=opts):
            if isinstance(msg, ResultMessage):
                result = msg
        if result is None or getattr(result, "is_error", False):
            raise RuntimeError(f"claude-sdk: {getattr(result, 'result', 'no result')}")
        self.spent += float(result.total_cost_usd or 0.0)
        usage = result.usage or {}
        cached = int(usage.get("cache_read_input_tokens", 0))
        CACHE["cached_in"] += cached
        CACHE["input"] += int(usage.get("input_tokens", 0)) + cached
        return result.result or ""


class AnthropicClient:
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def complete(
        self, system: str, user: str, max_tokens: int = 900, temperature: float = 1.0
    ) -> str:
        async with httpx.AsyncClient(timeout=25) as c:
            r = await c.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    # cache the identical system prompt every authoring call (~0.1x repeat cost)
                    "system": [
                        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
                    ],
                    "messages": [{"role": "user", "content": user}],
                },
            )
            r.raise_for_status()
            data = r.json()
            _u = data.get("usage", {})
            _cached = int(_u.get("cache_read_input_tokens", 0))
            CACHE["cached_in"] += _cached
            CACHE["input"] += int(_u.get("input_tokens", 0)) + _cached
            return "".join(
                b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
            )


def build_prompt(name: str, persona: str, summary: dict, current: dict) -> str:
    return (
        f"Your tribe '{name}' is {persona}\n"
        f"Right now: {summary['population']} organisms, avg energy {summary['avg_energy']}, "
        f"avg age {summary['avg_age']}, avg toxin in their cells {summary['avg_toxin']}, "
        f"avg free neighbor cells {summary['avg_free']}.\n"
        f"Your tribe's current DNA: {json.dumps(current)}\n"
        "Rewrite the DNA to express your temperament while keeping the tribe alive. "
        "Return the new genome."
    )


def _cond_from(d) -> Condition | None:
    if not isinstance(d, dict):
        return None
    var = d.get("variable")
    if var not in VARIABLES:
        return None
    op = d.get("op") if d.get("op") in OPS else ">"
    try:
        thr = int(d.get("threshold", 0))
    except (ValueError, TypeError):
        thr = 0
    return Condition(var, op, max(0, min(VAR_MAX[var], thr)))


def parse_genome(text: str, max_genes: int) -> Genome | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        raw = json.loads(text[start : end + 1])
    except Exception:
        return None
    regs = {}
    rr = raw.get("regulators", {}) if isinstance(raw.get("regulators"), dict) else {}
    for r in REGULATORS:
        try:
            regs[r] = max(0, min(100, int(rr.get(r, 50))))
        except (ValueError, TypeError):
            regs[r] = 50
    behaviors = []
    for b in raw.get("behaviors", [])[:max_genes]:
        if not isinstance(b, dict):
            continue
        c1 = _cond_from(b.get("cond1"))
        if c1 is None:
            continue
        c2 = _cond_from(b.get("cond2")) if b.get("cond2") else None
        verb = b.get("verb")
        if verb not in VERBS:
            continue
        target = b.get("target") if b.get("target") in TARGETS else "random"
        behaviors.append(Gene(c1, c2, verb, target))
    behaviors = dedupe(behaviors)
    if not behaviors:
        return None
    return Genome(regs, behaviors)


def parse_note(text: str) -> str:
    # the tribe's one-line dispatch — its own-voice rationale for this DNA rewrite
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ""
    try:
        raw = json.loads(text[start : end + 1])
    except Exception:
        return ""
    note = raw.get("note", "")
    return " ".join(str(note).split())[:140] if isinstance(note, str) else ""


async def author_genome(
    client: ModelClient, name: str, persona: str, summary: dict, current: dict, max_genes: int
) -> tuple[Genome | None, str]:
    text = await client.complete(SYSTEM, build_prompt(name, persona, summary, current))
    return parse_genome(text, max_genes), parse_note(text)
