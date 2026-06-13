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


# One resident CLI session per commander per match: process spawn (~7s) is paid once at
# the first command, then each call is just the model round-trip. The session's growing
# conversation IS the commander's match memory — and it's torn down at match end.
_sessions: dict[str, object] = {}


async def reset_sessions() -> None:
    global _sessions
    doomed, _sessions = _sessions, {}
    for client in doomed.values():
        try:
            await client.disconnect()
        except Exception:
            pass


async def _drop_session(key: str) -> None:
    client = _sessions.pop(key, None)
    if client is not None:
        try:
            await client.disconnect()
        except Exception:
            pass


def _read_result(result, tools):
    if result is None or getattr(result, "is_error", False):
        raise RuntimeError(f"claude-sdk: {getattr(result, 'result', 'no result')}")
    usage = result.usage or {}
    tin = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
    tout = int(usage.get("output_tokens", 0))
    text, calls = extract_call(result.result or "", tools)
    return text, calls, tin, tout, float(result.total_cost_usd or 0.0)


async def sdk_chat(
    ep: dict, system: str, transcript: list[dict], tools: list | None, session: str = ""
) -> tuple[str, list[dict], int, int, dict]:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, query

    prompt = "\n".join(m.get("text", "") for m in transcript)
    sysprompt = f"{system}\n{tool_instruction(tools)}"

    if session:
        client = _sessions.get(session)
        try:
            if client is None:
                # tools=[] already forbids an agentic loop, so no max_turns here — it would
                # count across the session and starve the second command
                opts = ClaudeAgentOptions(
                    model=ep["model"], system_prompt=sysprompt, allowed_tools=[], tools=[]
                )
                client = ClaudeSDKClient(opts)
                await client.connect()
                _sessions[session] = client
            await client.query(prompt)
            result = None
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    result = msg
            text, calls, tin, tout, cost = _read_result(result, tools)
        except Exception:
            await _drop_session(session)  # a broken session never gets reused
            raise
    else:
        opts = ClaudeAgentOptions(
            model=ep["model"],
            system_prompt=sysprompt,
            max_turns=1,
            allowed_tools=[],
            tools=[],
        )
        result = None
        async for msg in query(prompt=prompt, options=opts):
            if isinstance(msg, ResultMessage):
                result = msg
        text, calls, tin, tout, cost = _read_result(result, tools)

    out_ep = dict(ep)
    # the CLI reports the authoritative per-call cost (credit-side dollars at API rates)
    out_ep["flat_cost"] = cost
    return text, calls, tin, tout, out_ep
