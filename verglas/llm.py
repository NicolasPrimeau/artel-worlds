from __future__ import annotations


from . import env

from llmrouter import Request, Router, parse_json, pool

# Verglas's binding to the shared OpenRouter client: it resolves the pool from the environment and
# exposes the small functional surface the meeting code already calls.

_POOL = pool(env("POOL"))

ROUTER = Router(
    _POOL,
    concurrency=int(env("CONCURRENCY", "8")),
    cooldown=float(env("COOLDOWN", "8.0")),
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
