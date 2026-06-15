# Claude Agent SDK provider — subscription OAuth (CLAUDE_CODE_OAUTH_TOKEN) instead of an
# API key, so usage draws from the plan's monthly Agent SDK credit and PAUSES when it runs
# out rather than billing. The caller's failover chain treats a pause like any other dead
# endpoint and rolls to the next provider. JSON-by-instruction beats the SDK's structured
# output mode here: one turn, ~12x cheaper, and the command parser already tolerates junk.
from __future__ import annotations

import asyncio
import json
import os
import re

SDK_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
# the CLI cold-start (~20s on shared CPU) is a ONE-TIME cost: a resident session pays it
# once on connect, then every query is ~1s. Recycle a session after this many queries so its
# conversation can't grow unbounded across matches.
RECYCLE_AFTER = int(os.environ.get("PHALANX_SDK_RECYCLE", "60"))

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


# One resident CLI session per commander, kept WARM ACROSS MATCHES so the ~20s cold-start
# is paid once for the life of the process, not once per match. Each holds its client, the
# (shielded) connect task, and a query counter for recycling.
class _Session:
    __slots__ = ("client", "connect", "count")

    def __init__(self, client, connect):
        self.client = client
        self.connect = connect
        self.count = 0


_sessions: dict[str, _Session] = {}


async def _quiet_disconnect(client) -> None:
    try:
        await client.disconnect()
    except Exception:
        pass


async def reset_sessions() -> None:
    # full teardown — NOT called between matches (warm sessions must survive); only on
    # process-level reset. Per-match state is the conversation, bounded by RECYCLE_AFTER.
    global _sessions
    doomed, _sessions = _sessions, {}
    for s in doomed.values():
        await _quiet_disconnect(s.client)


async def _drop_session(key: str) -> None:
    s = _sessions.pop(key, None)
    if s is not None:
        await _quiet_disconnect(s.client)


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
        s = _sessions.get(session)
        try:
            if s is None:
                # tools=[] already forbids an agentic loop, so no max_turns here — it would
                # count across the session and starve the second command. connect() runs as
                # its own task so a command-deadline cancellation SHIELDS it: the cold-start
                # finishes in the background and the next command finds a warm session,
                # instead of every call re-paying (and re-cancelling) the ~20s spawn.
                opts = ClaudeAgentOptions(
                    model=ep["model"], system_prompt=sysprompt, allowed_tools=[], tools=[]
                )
                client = ClaudeSDKClient(opts)
                s = _Session(client, asyncio.ensure_future(client.connect()))
                _sessions[session] = s
            if not s.connect.done():
                # shield: if THIS command times out, the connect keeps running for next time.
                # CancelledError (the timeout) is a BaseException — it bypasses the except
                # Exception below, so a warming session is never torn down.
                await asyncio.shield(s.connect)
            if s.connect.cancelled() or s.connect.exception() is not None:
                raise RuntimeError("claude-sdk connect failed")
            await s.client.query(prompt)
            result = None
            async for msg in s.client.receive_response():
                if isinstance(msg, ResultMessage):
                    result = msg
            text, calls, tin, tout, cost = _read_result(result, tools)
            s.count += 1
            if s.count >= RECYCLE_AFTER:  # bound conversation growth: retire this session
                _sessions.pop(session, None)
                asyncio.ensure_future(_quiet_disconnect(s.client))
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
