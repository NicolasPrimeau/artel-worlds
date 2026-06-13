# Claude Agent SDK provider — subscription OAuth (CLAUDE_CODE_OAUTH_TOKEN) instead of an
# API key, so usage draws from the plan's monthly Agent SDK credit and PAUSES when it runs
# out rather than billing. The caller's failover chain treats a pause like any other dead
# endpoint and rolls to the next provider. JSON-by-instruction beats the SDK's structured
# output mode here: one turn, ~12x cheaper, and the command parser already tolerates junk.
from __future__ import annotations

import json
import os
import re

SDK_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")

_JSON_RE = re.compile(r"\{.*\}", re.S)


def tool_instruction(tools: list | None) -> str:
    if not tools:
        return "Reply with plain text only."
    t = tools[0]
    props = (t.get("schema") or t.get("input_schema") or {}).get("properties", {})
    fields = "\n".join(
        f'  "{k}": {v.get("type", "any")} — {v.get("description", "")}' for k, v in props.items()
    )
    return (
        f"Reply with ONLY one JSON object (no prose, no code fences): the arguments of your "
        f"'{t['name']}' call. Omit any field you don't use. Fields:\n{fields}"
    )


def extract_call(text: str, tools: list | None) -> tuple[str, list[dict]]:
    if not tools:
        return text, []
    m = _JSON_RE.search(text or "")
    if not m:
        return text, []
    try:
        args = json.loads(m.group(0))
    except (TypeError, ValueError):
        return text, []
    if not isinstance(args, dict):
        return text, []
    return "", [{"name": tools[0]["name"], "input": args}]


async def sdk_chat(
    ep: dict, system: str, transcript: list[dict], tools: list | None
) -> tuple[str, list[dict], int, int, dict]:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    prompt = "\n".join(m.get("text", "") for m in transcript)
    opts = ClaudeAgentOptions(
        model=ep["model"],
        system_prompt=f"{system}\n{tool_instruction(tools)}",
        max_turns=1,
        allowed_tools=[],
        tools=[],
    )
    result = None
    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, ResultMessage):
            result = msg
    if result is None or getattr(result, "is_error", False):
        raise RuntimeError(f"claude-sdk: {getattr(result, 'result', 'no result')}")
    usage = result.usage or {}
    tin = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
    tout = int(usage.get("output_tokens", 0))
    text, calls = extract_call(result.result or "", tools)
    out_ep = dict(ep)
    # the CLI reports the authoritative per-call cost (credit-side dollars at API rates)
    out_ep["flat_cost"] = float(result.total_cost_usd or 0.0)
    return text, calls, tin, tout, out_ep
