import asyncio
import json

import pytest

from app.utils.providers import registry
from app.utils.providers.openai_compat_llm import OpenAICompatLLM
from app.utils.providers.comfy_tts import ComfyUITTS, pick_tts_engine
from app.utils.providers.comfy_image import ComfyUIImageGen
from app.utils.providers.whisper_stt import FasterWhisperSTT
from app.utils.media.comfy_client import ComfyUIClient, ComfyUIUnavailable


# --- OpenAICompatLLM against a stub server ---

def aiohttp_serve_json(request, data):
    from aiohttp import web
    return web.json_response(data)


# lightweight fixture via threading server reused pattern:

@pytest.fixture
def local_loop_server():
    from tests.conftest_helpers import ThreadedAiohttpServer

    class App:
        def __init__(self):
            from aiohttp import web
            self.r = web.Application()
            self.captured = []
            r = self.r.router

            async def chat(request):
                from aiohttp import web as w
                b = await request.json()
                self.captured.append(b)
                return w.json_response({
                    "choices": [{"message": {"content": "hi from llm"}}],
                    "_auth": request.headers.get("Authorization", ""),
                    "_model": b.get("model"),
                })

            async def stats(request):
                return aiohttp_ok()

            r.add_post("/v1/chat/completions", chat)
            r.add_get("/system_stats", stats)

        def ok(self):
            from aiohttp import web as w
            return w.json_response({"ok": 1})

    return ThreadedAiohttpServer(App())


def aiohttp_ok():
    from aiohttp import web as w
    return w.json_response({"system": {}})


def test_llm_chat_posts_openai_shape(local_loop_server):
    url = local_loop_server.url + "/v1"
    llm = OpenAICompatLLM(base_url=url, api_key="sk-test", model="gemma4-abliterated")

    reply = asyncio.run(llm.chat([{"role": "user", "content": "ping"}]))

    assert reply == "hi from llm"
    sent = local_loop_server.obj.captured[0]
    assert sent["model"] == "gemma4-abliterated"
    assert sent["messages"][0]["content"] == "ping"


# --- ComfyUITTS end-to-end via stub /view ---

@pytest.fixture
def tts_workflow_dir(tmp_path, monkeypatch):
    d = tmp_path / "workflows" / "tts"
    d.mkdir(parents=True)
    (d / "myvoice.json").write_text(json.dumps({
        "3": {"class_type": "PrimitiveString", "inputs": {"value": ""}},
        "9": {"class_type": "SaveAudio", "inputs": {}},
    }))
    (d / "myvoice.meta.yaml").write_text("text_node_id: '3'\noutput_node_id: '9'\n")
    monkeypatch.setattr("app.utils.media.workflow_loader._workflows_dir_override", str(d.parent))
    yield d


def test_comfy_tts_generate_returns_file(tts_workflow_dir, threaded_comfy):
    client = ComfyUIClient(base_url=threaded_comfy.url, poll_interval_s=0.05,
                           timeout_s=10, cache_dir=str(threaded_comfy.obj.tmp))
    tts = ComfyUITTS(client)

    path = asyncio.run(tts.generate("привет", "akari"))

    assert path is not None
    from pathlib import Path
    p = Path(path)
    assert p.exists() and p.suffix == ".wav"


def _mk_threaded_comfy():
    from tests.conftest_helpers import ThreadedAiohttpServer

    class WFApp:
        def __init__(self):
            from aiohttp import web as w
            r = w.Application()
            self.tmp = None
            self.counter = {"n": 0}

            async def stats(request):
                return w.json_response({"system": {}})

            async def prompt(request):
                await request.json()
                self.counter["n"] += 1
                return w.json_response({"prompt_id": "p1"})

            async def history(request):
                n = self.counter["n"]
                if not hasattr(self, "polled"):
                    self.polled = 0
                self.polled += 1
                if self.polled < 2:
                    return w.json_response({})
                return w.json_response({"p1": {"outputs": {
                    "9": {"audios": [{"filename": "a.wav", "subfolder": "", "type": "output"}]}
                }}})

            async def view(request):
                name = request.query.get("filename", "")
                ctype = "audio/wav"
                return w.Response(body=b"RIFF-test-wav-bytes", content_type=ctype)

            r.router.add_get("/system_stats", stats)
            r.router.add_post("/prompt", prompt)
            r.router.add_get("/history/{pid}", history)
            r.router.add_get("/view", view)
            self.r = r

    return ThreadedAiohttpServer(WFApp())


@pytest.fixture
def threaded_comfy(tmp_path):
    box = _mk_threaded_comfy()
    box.obj.tmp = tmp_path
    yield box


def test_comfy_tts_unreachable_propagates(tmp_path):
    client = ComfyUIClient(base_url="http://127.0.0.1:59997", timeout_s=2,
                           cache_dir=str(tmp_path))
    tts = ComfyUITTS(client)
    with pytest.raises(ComfyUIUnavailable):
        asyncio.run(tts.generate("hello", "x"))


def test_pick_tts_engine_trims_dead_preserving_order(monkeypatch):
    monkeypatch.setattr(registry, "available", lambda kind: ["kokoro"])
    order = ["comfyui", "kokoro"]
    trimmed = pick_tts_engine(order)
    assert trimmed == ["kokoro"]


# --- Whisper STT config forwarding ---

class FakeWhisperModel:
    last_kwargs = {}

    def __init__(self, *args, **kwargs):
        FakeWhisperModel.last_kwargs = kwargs


def test_whisper_stt_forwards_config_paths(tmp_path, monkeypatch):
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir(parents=True)
    (cfg_root / "llm.yaml").write_text(
        'stt:\n'
        f'  whisper_model_path: "{tmp_path / "weights"}"\n'
        '  device: cpu\n'
        '  compute_type: int8\n'
    )
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg_root))
    from app.utils import xconfig
    xconfig.reset_cache()

    stt = FasterWhisperSTT(model_cls=FakeWhisperModel)
    stt._load()  # lazy by design: constructed-but-not-loaded until first transcribe

    assert FakeWhisperModel.last_kwargs["model_size_or_path"] == str(tmp_path / "weights")
    assert FakeWhisperModel.last_kwargs["device"] == "cpu"


# --- image gen returns first image bytes ---

@pytest.fixture
def img_workflow_dir(tmp_path, monkeypatch):
    d = tmp_path / "workflows" / "image"
    d.mkdir(parents=True)
    (d / "painter.json").write_text(json.dumps({
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "9": {"class_type": "SaveImage", "inputs": {}},
    }))
    monkeypatch.setattr("app.utils.media.workflow_loader._workflows_dir_override", str(d.parent))
    yield d


def test_image_gen_returns_bytes(img_workflow_dir, img_server):
    img = ComfyUIImageGen(ComfyUIClient(base_url=img_server.url, poll_interval_s=0.05,
                                        timeout_s=10, cache_dir=str(tmp_path_factory := __import__("tempfile").gettempdir())))
    data = asyncio.run(img.generate("a cat"))
    assert data.startswith(b"\x89PNG")


class _ImgServerBox:
    pass


@pytest.fixture
def img_server():
    from tests.conftest_helpers import ThreadedAiohttpServer

    class ImgApp:
        def __init__(self):
            from aiohttp import web as w
            r = w.Application()
            self.polled = {"n": 0}

            async def stats(request):
                return w.json_response({"system": {}})

            async def prompt(request):
                await request.json()
                return w.json_response({"prompt_id": "pi"})

            async def history(request):
                self.polled["n"] += 1
                if self.polled["n"] < 2:
                    return w.json_response({})
                return w.json_response({"pi": {"outputs": {
                    "9": {"images": [{"filename": "x.png", "subfolder": "", "type": "output"}]}
                }}})

            async def view(request):
                return w.Response(body=b"\x89PNG-fake", content_type="image/png")

            r.router.add_get("/system_stats", stats)
            r.router.add_post("/prompt", prompt)
            r.router.add_get("/history/{pid}", history)
            r.router.add_get("/view", view)
            self.r = r

    return ThreadedAiohttpServer(ImgApp())
