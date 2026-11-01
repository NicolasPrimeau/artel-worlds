from __future__ import annotations

import os

from llmrouter import Request, Router, parse_json, pool

# Phalanx's binding to the shared OpenRouter client: act() for a forced tool call (the
# standing-orders decision) and complete() for plain text (huddle plans, post-match lessons).
# The client only accepts free models, so a commander cannot spend money.

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


_POOL = pool(_env("POOL"))
REASONING = _env("REASONING", "none")

ROUTER = Router(
    _POOL,
    concurrency=int(_env("CONCURRENCY", "8")),
    cooldown=float(_env("COOLDOWN", "8.0")),
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
            max_tokens=max_tokens,
            reasoning_effort=REASONING,
        )
    )
