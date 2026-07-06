from __future__ import annotations

import os

from llmrouter import Request, Router, build_models, parse_json

# Phalanx's binding to the shared llmrouter. The generic Router owns round-robin, 429-skip,
# capability filtering and per-model telemetry; this module resolves Phalanx's keys/pool from
# the environment and exposes the two calls the commander needs: act() for a forced tool call
# (the standing-orders decision) and complete() for plain text (huddle plans, post-match
# lessons). Phalanx routes ONLY over free, fast, capable providers — no per-token billing,
# no reasoning stalls — so a commander stays sharp without draining a budget.

__all__ = [
    "ROUTER",
    "act",
    "complete",
    "describe",
    "enabled",
    "metrics",
    "openai_tools",
    "parse_json",
    "spend",
]


def _env(key: str, default: str = "") -> str:
    return os.environ.get(f"PHALANX_{key}", default)


# Phalanx already ships PHALANX_LLM_KEY (groq) and PHALANX_LLM2_KEY (gemini); the other free
# tiers reuse Verglas's key names so one deploy can copy the same provider secrets across worlds.
_KEYS = {
    "groq": _env("LLM_KEY"),
    "cerebras": _env("CEREBRAS_KEY"),
    "sambanova": _env("SAMBANOVA_KEY"),
    "nvidia": _env("NVIDIA_KEY"),
    "gemini": _env("LLM2_KEY"),
}

# The pool the router cycles through, each entry "provider:model". Spread across every free tier
# that has a key so concurrent per-tank calls land in independent rate buckets. Override with
# PHALANX_POOL. Every model here supports tool calling (the command decision is a forced tool).
# Gemini is deliberately EXCLUDED: it's an AI Studio key whose Google project may have billing
# enabled (unlike Groq/Cerebras, which are card-free free tiers) — keep phalanx unambiguously $0.
# Re-add "gemini:gemini-2.5-flash" via PHALANX_POOL if that project is confirmed free-tier.
_DEFAULT_POOL = [
    "groq:openai/gpt-oss-120b",
    "groq:llama-3.3-70b-versatile",
    "groq:qwen/qwen3-32b",
    "groq:openai/gpt-oss-20b",
    "cerebras:gpt-oss-120b",
    "cerebras:zai-glm-4.7",
    "sambanova:gpt-oss-120b",
    "sambanova:Meta-Llama-3.3-70B-Instruct",
    "nvidia:meta/llama-3.3-70b-instruct",
]
_POOL = [s for s in _env("POOL", ",".join(_DEFAULT_POOL)).split(",") if s.strip()]

# A tactical commander needs a real model, so the router floor is BALANCED — capable (gpt-oss-120b,
# llama-3.3-70b) plus the balanced tier (qwen3-32b, gpt-oss-20b, glm-4.7), never the fast/flash-lite
# tier. Balanced is included so groq's per-model rate buckets (gpt-oss-20b, qwen3-32b) and cerebras
# glm-4.7 share the load — without them a 2-provider capable-only pool 429s hard. Override with
# PHALANX_MIN_GRADE=capable to tighten again. Every eligible model still supports tool calling.
MIN_GRADE = _env("MIN_GRADE", "balanced")

# An arena turn is a single tool decision against a tight max_tokens — chain-of-thought is not
# needed and, on gemini-2.5, hidden thinking eats the whole budget and returns an empty command.
# Declare reasoning OFF; the router maps it to each provider's supported knob (or drops it).
REASONING = _env("REASONING", "none")


def _shape(user: str, model: str) -> str:
    # Qwen3 reasons unless told not to; centralise that quirk in the router, not the agent code.
    return user + ("\n/no_think" if "qwen" in model.lower() else "")


ROUTER = Router(
    build_models(_POOL, _KEYS),
    concurrency=int(_env("CONCURRENCY", "8")),
    cooldown=float(_env("COOLDOWN", "8.0")),
    shaper=_shape,
)


def openai_tools(toolset: list) -> list:
    # Phalanx tool schema ({name, description, schema}) -> OpenAI function-tool shape the router speaks.
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["schema"],
            },
        }
        for t in toolset
    ]


def enabled() -> bool:
    return ROUTER.enabled()


def metrics() -> list[dict]:
    return ROUTER.metrics()


def spend() -> dict:
    return ROUTER.spend


def describe() -> str:
    return ROUTER.describe()


async def act(
    system: str,
    user: str,
    toolset: list,
    *,
    temperature: float = 0.7,
    timeout: float = 12.0,
    max_tokens: int = 320,
) -> dict | None:
    # one forced tool call — returns {"name", "args"} or None if no model produced one (the caller
    # then leaves the tank on its deterministic motor). tool_choice="required" preserves the old
    # force_act: a capable model must answer with the command tool, not prose.
    return await ROUTER.act(
        Request(
            system=system,
            user=user,
            temperature=temperature,
            timeout=timeout,
            requires_tools=True,
            min_grade=MIN_GRADE,
            max_tokens=max_tokens,
            tools=openai_tools(toolset),
            tool_choice="required",
            reasoning_effort=REASONING,
        )
    )


async def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = 90,
    temperature: float = 0.7,
    timeout: float = 12.0,
) -> str:
    return await ROUTER.complete(
        Request(
            system=system,
            user=user,
            temperature=temperature,
            timeout=timeout,
            min_grade=MIN_GRADE,
            max_tokens=max_tokens,
            reasoning_effort=REASONING,
        )
    )
