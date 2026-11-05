"""Branching in AIFactory: external llm.yaml backend vs embedded llama-server."""

import asyncio

import pytest

from app.utils import xconfig
from app.utils.ai_clients.ai_factory import AIFactory
from app.utils.ai_clients.providers.local_provider import LocalProvider


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg))
    xconfig.reset_cache()
    yield cfg


def test_external_backend_used_when_set(config_root):
    (config_root / "llm.yaml").write_text(
        "backend: http://127.0.0.1:8080/\n"
        "model: gemma4-abliterated\n"
        "api_key: ollama\n"
        "temperature: 0.85\n"
        "repeat_penalty: 1.05\n"
        'stop_tokens: ["<end_of_turn>"]\n'
    )

    provider = AIFactory.get_provider("Local LLM")

    assert isinstance(provider, LocalProvider)
    # Trailing slash tolerated, /v1 appended once (same contract as OpenAICompatLLM)
    assert provider.base_url == "http://127.0.0.1:8080/v1"
    assert provider.model == "gemma4-abliterated"
    assert provider.api_key == "ollama"
    assert provider.sampling_defaults["temperature"] == 0.85
    assert provider.sampling_defaults["repeat_penalty"] == 1.05
    assert provider.sampling_defaults["stop"] == ["<end_of_turn>"]


def test_empty_backend_keeps_embedded_launcher(config_root):
    xconfig.load("llm")  # seed defaults; backend is "" by default

    provider = AIFactory.get_provider("Local LLM")

    assert isinstance(provider, LocalProvider)
    assert provider.base_url == "http://127.0.0.1:48596/v1"
    assert provider.model == "local-model"
    assert provider.sampling_defaults == {}


def test_external_only_v1_suffix_not_duplicated(config_root):
    (config_root / "llm.yaml").write_text("backend: http://127.0.0.1:9999/v1\n")

    provider = AIFactory.get_provider("Local LLM")

    assert provider.base_url == "http://127.0.0.1:9999/v1"


def test_local_server_manager_skips_launch_on_external_backend(config_root, monkeypatch):
    (config_root / "llm.yaml").write_text("backend: http://127.0.0.1:8080\n")
    from app.utils.ai_clients.local_server_manager import LocalServerManager

    mgr = LocalServerManager(ui=None)
    touched = {"port_check": False, "spawn": False}
    monkeypatch.setattr(mgr, "is_server_already_running", lambda: touched.__setitem__("port_check", True))
    monkeypatch.setattr(mgr, "start_server_async", lambda: touched.__setitem__("spawn", True))

    asyncio.run(mgr.ensure_server_running())

    assert mgr.model_loaded is True
    assert touched == {"port_check": False, "spawn": False}


def test_external_provider_requests_hit_stub_server(config_root):
    from tests.conftest_helpers import ThreadedAiohttpServer

    class LLMApp:
        def __init__(self):
            from aiohttp import web as w
            r = w.Application()
            self.captured = []

            async def chat(request):
                b = await request.json()
                self.captured.append(b)
                return w.json_response({
                    "id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                    "model": b.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "ext-ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

            r.router.add_post("/v1/chat/completions", chat)
            self.r = r

    box = ThreadedAiohttpServer(LLMApp())
    try:
        (config_root / "llm.yaml").write_text(
            f"backend: {box.url}\n"
            "model: gemma4-abliterated\n"
            "temperature: 0.85\n"
            "repeat_penalty: 1.05\n"
            'stop_tokens: ["<end_of_turn>"]\n'
        )
        provider = AIFactory.get_provider("Local LLM")
        messages = [{"role": "user", "content": "hi"}]

        result = asyncio.run(provider.generate(messages))

        assert result["content"] == "ext-ok"
        sent = box.obj.captured[0]
        assert sent["model"] == "gemma4-abliterated"
        assert sent["temperature"] == 0.85
        assert sent["repeat_penalty"] == 1.05
        assert sent["stop"] == ["<end_of_turn>"]

        # Explicit call kwargs win over llm.yaml defaults
        asyncio.run(provider.generate(messages, temperature=0.3))
        assert box.obj.captured[1]["temperature"] == 0.3
        assert box.obj.captured[1]["stop"] == ["<end_of_turn>"]
    finally:
        box.stop()


def test_external_provider_streams_from_stub_server(config_root):
    from tests.conftest_helpers import ThreadedAiohttpServer

    class LLMApp:
        def __init__(self):
            from aiohttp import web as w
            r = w.Application()
            self.captured = []

            async def chat(request):
                b = await request.json()
                self.captured.append(b)

                async def sse():
                    for piece in ("hel", "lo"):
                        delta = '{"content":"%s"}' % piece
                        yield 'data: {"id":"c1","choices":[{"index":0,"delta":%s,"finish_reason":null}]}\n\n' % delta
                    yield 'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    yield "data: [DONE]\n\n"

                resp = w.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                async for frame in sse():
                    await resp.write(frame.encode())
                await resp.write_eof()

            r.router.add_post("/v1/chat/completions", chat)
            self.r = r

    box = ThreadedAiohttpServer(LLMApp())
    try:
        (config_root / "llm.yaml").write_text(
            f"backend: {box.url}\n"
            "model: gemma4-abliterated\n"
        )
        provider = AIFactory.get_provider("Local LLM")

        async def collect():
            return [chunk async for chunk in provider.generate_stream([{"role": "user", "content": "hi"}])]

        chunks = asyncio.run(collect())

        assert "".join(chunks) == "hello"
        assert box.obj.captured[0]["model"] == "gemma4-abliterated"
        assert box.obj.captured[0]["stream"] is True
    finally:
        box.stop()
