import asyncio

import httpx
import pytest

from llmrouter import Request, Router, pool


def _router(handler, **kwargs):
    router = Router(["a/one:free", "b/two:free"], key="k", **kwargs)
    real = httpx.AsyncClient

    def client(**opts):
        return real(transport=httpx.MockTransport(handler), **opts)

    return router, client


def _reply(message, model="b/two:free", usage=None):
    return httpx.Response(
        200, json={"model": model, "choices": [{"message": message}], "usage": usage or {}}
    )


def test_paid_models_are_refused():
    with pytest.raises(ValueError):
        Router(["openai/gpt-5"], key="k")


def test_no_key_means_disabled_and_no_call():
    router = Router(key="")
    assert not router.enabled()
    assert asyncio.run(router.complete(Request(system="s", user="u"))) == ""


def test_sends_fallback_list_and_records_the_model_that_answered(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(httpx.Response(200, content=request.content).json())
        return _reply({"content": "hi"}, usage={"prompt_tokens": 2, "completion_tokens": 1})

    router, client = _router(handler)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    out, model = asyncio.run(router.complete_m(Request(system="s", user="u", max_tokens=9)))
    assert (out, model) == ("hi", "b/two:free")
    assert seen["models"] == ["a/one:free", "b/two:free"]
    assert seen["max_tokens"] == 9
    assert router.spend["in"] == 2
    assert [m["ok"] for m in router.metrics() if m["model"] == "b/two:free"] == [1]


def test_tool_requests_require_a_provider_that_supports_them(monkeypatch):
    seen = {}

    def handler(request):
        seen.update(httpx.Response(200, content=request.content).json())
        call = {"function": {"name": "move", "arguments": '{"dir": "north"}'}}
        return _reply({"tool_calls": [call]})

    router, client = _router(handler)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    request = Request(system="s", user="u", tools=[{}], tool_choice="required")
    assert asyncio.run(router.act(request)) == {"name": "move", "args": {"dir": "north"}}
    assert seen["provider"] == {"require_parameters": True}


def test_a_429_pauses_calls_instead_of_hammering(monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "30"})

    router, client = _router(handler)
    monkeypatch.setattr(httpx, "AsyncClient", client)
    assert asyncio.run(router.complete(Request(system="s", user="u"))) == ""
    assert asyncio.run(router.complete(Request(system="s", user="u"))) == ""
    assert len(calls) == 1
    assert router.metrics()[0]["cooling"]


def test_pool_falls_back_to_defaults():
    assert pool("x:free, y:free") == ["x:free", "y:free"]
    assert pool("") == pool(None)
