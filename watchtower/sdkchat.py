# Claude Agent SDK provider — subscription OAuth (CLAUDE_CODE_OAUTH_TOKEN), drawing the
# plan's monthly Agent SDK credit; usage PAUSES at exhaustion and the failover chain rolls
# to the next provider. Watchtower's responder is a multi-round tool loop, so each round
# flattens the transcript to text and asks for ONE tool pick as plain JSON — the loop
# itself stays in world code, exactly as with the HTTP providers.
from __future__ import annotations

import json
import os
import re

SDK_TOKEN = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")

_JSON_RE = re.compile(r"\{.*\}", re.S)
_ids = iter(range(1, 10_000_000))


def tools_catalog(tools: list | None) -> str:
    if not tools:
        return "Reply with plain text only."
    lines = []
    for t in tools:
        props = (t.get("schema") or {}).get("properties", {})
        args = ", ".join(f'"{k}": {v.get("type", "any")}' for k, v in props.items())
        lines.append(f"- {t['name']}({args}): {t['description']}")
    return (
        "To act, reply with ONLY one JSON object (no prose, no code fences): "
        '{"tool": "<name>", "args": {...}} picking ONE tool from the catalog below. '
        "When you are completely done and need no tool, reply with plain text instead.\n"
        "Tools:\n" + "\n".join(lines)
    )


def flatten(transcript: list[dict]) -> str:
    lines = []
    for e in transcript:
        if e["role"] == "user":
            lines.append(e.get("text", ""))
        elif e["role"] == "assistant":
            if e.get("text"):
                lines.append(f"[you said] {e['text']}")
            for c in e.get("calls", []):
                lines.append(f"[you called] {c['name']}({json.dumps(c.get('input', {}))})")
        else:
            for r in e.get("results", []):
                lines.append(f"[tool result] {r.get('output', '')}")
    return "\n".join(lines)


def extract_tool_call(text: str, tools: list | None) -> tuple[str, list[dict]]:
    if not tools:
        return text, []
    m = _JSON_RE.search(text or "")
    if not m:
        return text, []
    try:
        obj = json.loads(m.group(0))
    except (TypeError, ValueError):
        return text, []
    name = obj.get("tool") if isinstance(obj, dict) else None
    if name not in {t["name"] for t in tools}:
        return text, []
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return "", [{"id": f"sdk{next(_ids)}", "name": name, "input": args}]


# One resident CLI session per INCIDENT: the SDK conversation IS the incident transcript,
# so each round sends only what's new (tool results, state notes) instead of re-flattening
# the whole history — spawn paid once, tokens linear instead of quadratic. A broken session
# is dropped; the next round rebuilds from the full transcript.
_sessions: dict[str, tuple[object, int]] = {}


async def drop_session(key: str) -> None:
    client, _ = _sessions.pop(key, (None, 0))
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
    text, calls = extract_tool_call(result.result or "", tools)
    return text, calls, tin, tout, float(result.total_cost_usd or 0.0)


async def sdk_chat(
    ep: dict, system: str, transcript: list[dict], tools: list | None, session: str = ""
) -> tuple[str, list[dict], int, int, dict]:
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, query

    sysprompt = f"{system}\n{tools_catalog(tools)}"

    if session:
        client, sent = _sessions.get(session, (None, 0))
        try:
            if client is None:
                # tools=[] forbids an agentic loop; no max_turns — it counts across the
                # session and would starve every round after the first
                opts = ClaudeAgentOptions(
                    model=ep["model"], system_prompt=sysprompt, allowed_tools=[], tools=[]
                )
                client = ClaudeSDKClient(opts)
                await client.connect()
                sent = 0
            await client.query(flatten(transcript[sent:]))
            result = None
            async for msg in client.receive_response():
                if isinstance(msg, ResultMessage):
                    result = msg
            text, calls, tin, tout, cost = _read_result(result, tools)
            _sessions[session] = (client, len(transcript))
        except Exception:
            await drop_session(session)
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
        async for msg in query(prompt=flatten(transcript), options=opts):
            if isinstance(msg, ResultMessage):
                result = msg
        text, calls, tin, tout, cost = _read_result(result, tools)

    out_ep = dict(ep)
    out_ep["flat_cost"] = cost
    return text, calls, tin, tout, out_ep
