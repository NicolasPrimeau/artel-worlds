import asyncio

from llmrouter.models import Model
from llmrouter.router import Request, Router


def _model(provider="groq", model="llama-3.1-8b-instant"):
    return Model(
        provider=provider, model=model, url="http://x/v1", key="k", tools=True, grade="fast"
    )


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def json(self):
        return self._p

    @property
    def request(self):
        return None


class _FakeHTTP:
    def __init__(self):
        self.last_json = None

    async def post(self, url, headers=None, json=None):
        self.last_json = json
        return _FakeResp(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )


def test_max_tokens_uses_plain_key_for_openai_compatible():
    r = Router([_model()])
    h = _FakeHTTP()
    asyncio.run(r._call(h, r.models[0], Request(system="s", user="u", max_tokens=100)))
    assert h.last_json["max_tokens"] == 100
    assert "max_completion_tokens" not in h.last_json


def test_max_tokens_uses_completion_key_for_gpt5():
    r = Router([_model()])
    h = _FakeHTTP()
    m5 = _model(provider="openai", model="gpt-5-nano")
    asyncio.run(r._call(h, m5, Request(system="s", user="u", max_tokens=200)))
    assert h.last_json["max_completion_tokens"] == 200
    assert "max_tokens" not in h.last_json
    assert h.last_json["temperature"] == 1  # gpt-5 forces default temp
    assert h.last_json["reasoning_effort"] == "minimal"


def test_no_token_cap_when_unset():
    r = Router([_model()])
    h = _FakeHTTP()
    asyncio.run(r._call(h, r.models[0], Request(system="s", user="u")))
    assert "max_tokens" not in h.last_json
    assert "max_completion_tokens" not in h.last_json


def _reasoning(model, provider="groq", **req):
    r = Router([_model()])
    h = _FakeHTTP()
    asyncio.run(
        r._call(h, _model(provider=provider, model=model), Request(system="s", user="u", **req))
    )
    return h.last_json.get("reasoning_effort", "<unset>")


def test_reasoning_effort_maps_per_provider():
    # gpt-oss accepts only low/medium/high — an off/none request clamps to its floor, never 400s
    assert _reasoning("openai/gpt-oss-120b", reasoning_effort="none") == "low"
    assert _reasoning("openai/gpt-oss-120b") == "low"  # default stays minimal
    # llama/qwen/glm expose no knob — the field must be OMITTED even when asked, or the provider 400s
    assert _reasoning("llama-3.3-70b-versatile", reasoning_effort="none") == "<unset>"
    assert _reasoning("zai-glm-4.7", reasoning_effort="none") == "<unset>"
    # gemini-2.5 thinks by default: off for tool calls (or when asked), untouched for plain text
    assert _reasoning("gemini-2.5-flash", reasoning_effort="none", tools=[{"x": 1}]) == "none"
    assert _reasoning("gemini-2.5-flash", tools=[{"x": 1}]) == "none"
    assert _reasoning("gemini-2.5-flash") == "<unset>"  # plain-text caller keeps the model default


def test_tool_choice_defaults_to_auto_and_honors_override():
    r = Router([_model()])
    h = _FakeHTTP()
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    asyncio.run(r._call(h, r.models[0], Request(system="s", user="u", tools=tools)))
    assert h.last_json["tool_choice"] == "auto"
    asyncio.run(
        r._call(h, r.models[0], Request(system="s", user="u", tools=tools, tool_choice="required"))
    )
    assert h.last_json["tool_choice"] == "required"


def test_run_stops_at_deadline(monkeypatch):
    r = Router([_model()])
    calls = {"n": 0}

    async def _slow_miss(h, model, req):
        calls["n"] += 1
        await asyncio.sleep(0.03)
        return {"content": None}  # a miss -> _text returns None -> next attempt

    monkeypatch.setattr(r, "_call", _slow_miss)
    res, _ = asyncio.run(r._run(Request(system="s", user="u", deadline=0.02), r._text))
    assert res is None
    assert calls["n"] <= 2  # deadline cut it well below tries = max(4, 2) = 4


def test_deadline_none_runs_full_tries(monkeypatch):
    r = Router([_model()])
    calls = {"n": 0}

    async def _fast_miss(h, model, req):
        calls["n"] += 1
        return {"content": None}

    monkeypatch.setattr(r, "_call", _fast_miss)
    asyncio.run(r._run(Request(system="s", user="u"), r._text))
    assert calls["n"] == max(4, len(r.models) + 1)  # no deadline -> exhausts the retry budget
