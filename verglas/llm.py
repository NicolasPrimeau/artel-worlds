from __future__ import annotations


from . import env

from llmrouter import Request, Router, build_models, parse_json

# Verglas's binding to the shared llmrouter. The generic Router owns round-robin, 429-skip, capability
# filtering and telemetry; this module just resolves Verglas's keys/pool from the environment and exposes
# the small functional surface the meeting code already calls. Verglas doesn't use tool calling — it parses
# JSON out of plain completions — so requests leave requires_tools=False.

_KEYS = {
    "groq": env("LLM_KEY"),
    "cerebras": env("CEREBRAS_KEY"),
    "sambanova": env("SAMBANOVA_KEY"),
    "nvidia": env("NVIDIA_KEY"),
    "gemini": env("LLM2_KEY"),
}

# the pool the router cycles through, each entry "provider:model". The default spreads across every free
# tier that has a key so concurrent calls land in independent rate buckets. Override with VERGLAS_POOL.
_DEFAULT_POOL = [
    "groq:openai/gpt-oss-120b",
    "groq:llama-3.3-70b-versatile",
    "groq:llama-3.1-8b-instant",
    "groq:qwen/qwen3-32b",
    "groq:openai/gpt-oss-20b",
    "groq:meta-llama/llama-4-scout-17b-16e-instruct",
    "cerebras:gpt-oss-120b",
    "cerebras:zai-glm-4.7",
    "sambanova:gpt-oss-120b",
    "sambanova:Meta-Llama-3.3-70B-Instruct",
    "nvidia:meta/llama-3.3-70b-instruct",
    "gemini:gemini-2.5-flash",
    "gemini:gemini-flash-lite-latest",
]
_POOL = [s for s in env("POOL", ",".join(_DEFAULT_POOL)).split(",") if s.strip()]


def _shape(user: str, model: str) -> str:
    # Qwen3 reasons unless told not to; centralise that quirk in the router rather than the agent code.
    return user + ("\n/no_think" if "qwen" in model.lower() else "")


ROUTER = Router(
    build_models(_POOL, _KEYS),
    concurrency=int(env("CONCURRENCY", "8")),
    cooldown=float(env("COOLDOWN", "8.0")),
    cost_in_per_m=float(env("COST_IN_PER_M", "0.15")),
    cost_out_per_m=float(env("COST_OUT_PER_M", "0.60")),
    shaper=_shape,
)

SPEND = ROUTER.spend  # the server persists/reads this dict; same object, so it stays in sync
POOL_DESC = ROUTER.describe()


def enabled() -> bool:
    return ROUTER.enabled()


def metrics() -> list[dict]:
    return ROUTER.metrics()


async def complete(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.7,
    timeout: float = 16.0,
) -> str:
    # model is ignored — the router owns model choice (agents are decoupled). Kept for call-site compat.
    return await ROUTER.complete(
        Request(system=system, user=user, temperature=temperature, timeout=timeout)
    )


async def complete_many(jobs: list[tuple[str, str]], temperature: float = 0.7) -> list[str]:
    return await ROUTER.complete_many(
        [Request(system=s, user=u, temperature=temperature) for s, u in jobs]
    )


async def complete_m(
    system: str, user: str, temperature: float = 0.7, timeout: float = 16.0
) -> tuple:
    # like complete(), but returns (text, model_id) — so a line can be attributed to the model that spoke it
    return await ROUTER.complete_m(
        Request(system=system, user=user, temperature=temperature, timeout=timeout)
    )


async def complete_many_m(jobs: list[tuple[str, str]], temperature: float = 0.7) -> list[tuple]:
    return await ROUTER.complete_many_m(
        [Request(system=s, user=u, temperature=temperature) for s, u in jobs]
    )


async def act_many(reqs: list) -> list:
    # batch of tool-calling decisions; each returns {"name", "args"} or None (the caller falls back)
    return await ROUTER.act_many(reqs)


async def act_many_m(reqs: list) -> list:
    # like act_many(), but each entry is (action|None, model_id|None)
    return await ROUTER.act_many_m(reqs)


__all__ = [
    "POOL_DESC",
    "ROUTER",
    "SPEND",
    "act_many",
    "act_many_m",
    "complete",
    "complete_m",
    "complete_many",
    "complete_many_m",
    "enabled",
    "metrics",
    "parse_json",
]
