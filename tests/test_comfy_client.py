import asyncio
import json
import threading

import aiohttp
import aiohttp.test_utils as atu
from aiohttp import web, web_runner
import pytest

from app.utils.media.comfy_client import ComfyUIClient, ComfyUIUnavailable


PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-image-data"
WAV_BYTES = b"RIFFfakedata-wav"


class _StubApp(atu.Application):
    """Minimal ComfyUI API stand-in."""

    def __init__(self):
        super().__init__()
        self.router.add_get("/system_stats", self.stats)
        self.router.add_post("/prompt", self.prompt)
        self.router.add_get("/history/{pid}", self.history)
        self.router.add_get("/view", self.view)
        self._prompt_count = 0
        self.captured_bodies = []

    async def stats(self, request):
        return web.Response(text='{"system": {}}', content_type="application/json")

    async def prompt(self, request):
        self._prompt_count += 1
        body = await request.json()
        self.captured_bodies.append(body)
        return web.Response(text=json.dumps({"prompt_id": "job-1"}), content_type="application/json")

    async def history(self, request):
        pid = request.match_info["pid"]
        if self._prompt_count < 1 or len(self.captured_bodies) == 0:
            return web.Response(text="{}", content_type="application/json")
        # first two polls: empty; then produce outputs
        if not getattr(self, "_done", False):
            self._done = True
            return web.Response(text="{}", content_type="application/json")
        out = {
            pid: {
                "outputs": {
                    "9": {
                        "audios": [{"filename": "a.wav", "subfolder": "", "type": "output"}],
                        "images": [{"filename": "i.png", "subfolder": "sub", "type": "output"}],
                    }
                }
            }
        }
        return web.Response(text=json.dumps(out), content_type="application/json")

    async def view(self, request):
        name = request.query.get("filename", "")
        data = WAV_BYTES if name.endswith(".wav") else PNG_BYTES
        ctype = "audio/wav" if name.endswith(".wav") else "image/png"
        return web.Response(body=data, content_type=ctype)


@pytest.fixture
def server():
    async def _start():
        app = _StubApp()
        runner = web_runner.AppRunner(app)
        await runner.setup()
        site = web_runner.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        return runner, f"http://127.0.0.1:{port}", app

    loop = asyncio.new_event_loop()
    started = threading.Event()
    box = {}

    def run():
        runner, url, app = loop.run_until_complete(_start())
        box.update(runner=runner, url=url, app=app, loop=loop)
        started.set()
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    started.wait(timeout=5)

    yield box

    loop.call_soon_threadsafe(box["loop"].stop)


def test_run_full_cycle_returns_artifact_files(server):
    client = ComfyUIClient(base_url=server["url"], poll_interval_s=0.05, timeout_s=10)

    result = asyncio.run(client.run(_wf(), node_inputs={"3": {"value": "hi"}}))

    assert len(result["audio"]) == 1 and result["audio"][0].exists()
    assert result["audio"][0].read_bytes() == WAV_BYTES
    assert len(result["images"]) == 1 and result["images"][0].read_bytes() == PNG_BYTES
    sent = server["app"].captured_bodies[0]["prompt"]["3"]["inputs"]["value"]
    assert sent == "hi"


def _wf():
    return {
        "3": {"class_type": "PrimitiveString", "inputs": {"value": ""}},
        "9": {"class_type": "SaveAudio", "inputs": {}},
    }


def test_history_no_outputs_returns_empty_result(server):
    class NoOut(_StubApp):
        async def history(self, request):
            self._prompt_count = 99
            if not hasattr(self, "n"):
                self.n = 0
            self.n += 1
            if self.n < 2:
                return web.Response(text="{}", content_type="application/json")
            out = {request.match_info["pid"]: {"outputs": {}}}
            return web.Response(text=json.dumps(out), content_type="application/json")

    import aiohttp

    async def _start():
        app = NoOut()
        runner = web_runner.AppRunner(app)
        await runner.setup()
        site = web_runner.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        return f"http://127.0.0.1:{port}"

    fut = asyncio.run_coroutine_threadsafe(_start(), server["loop"])
    url = fut.result(timeout=5)
    client = ComfyUIClient(base_url=url, poll_interval_s=0.05, timeout_s=10)

    result = asyncio.run(client.run(_wf()))
    assert result == {"audio": [], "images": [], "video": []}


def test_server_down_raises_unavailable(tmp_path):
    client = ComfyUIClient(base_url="http://127.0.0.1:59999", timeout_s=2)
    with pytest.raises(ComfyUIUnavailable):
        asyncio.run(client.run(_wf(), cache_dir=str(tmp_path)))


def test_ping_false_when_dead(server):
    client = ComfyUIClient(base_url="http://127.0.0.1:59998")
    ok = asyncio.run(client.ping())
    assert ok is False
    assert client.healthy is False


def test_base_url_from_config(server, tmp_path, monkeypatch):
    cfg_root = tmp_path / "cfg"
    (cfg_root).mkdir(parents=True)
    (cfg_root / "media.yaml").write_text(
        f"comfyui:\n  enabled: true\n  base_url: {server['url']}\n"
        "  timeout_s: 5\n  poll_interval_s: 0.05\n"
        '  cache_dir: app/cache/comfy\nworkflows_dir: app/data/workflows\n'
    )
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg_root))
    from app.utils import xconfig
    xconfig.reset_cache()

    client = ComfyUIClient()

    assert client.base_url == server["url"]
