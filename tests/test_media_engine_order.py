"""Phase-B wiring: media.yaml tts_engine_order / image_provider_order.

Covers chain construction, the fallback walkers, real media.yaml reads via
xconfig, the ImageGenerator dispatch loop, and the TTSWorker chain runner.
"""

import asyncio
import logging
import types

import pytest

from app.utils.providers.comfy_tts import (
    arun_fallback_chain,
    configured_tts_chain,
    run_fallback_chain,
    tts_engine_chain,
)
from app.utils.image_generator import (
    PROVIDER_A1111,
    PROVIDER_COMFY_WF,
    PROVIDER_DALLE,
    PROVIDER_FLUX,
    PROVIDER_NOVELAI,
    image_provider_chain,
)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def media_config(tmp_path, monkeypatch):
    """Points SOW_CONFIG_DIR at a temp root; write() replaces media.yaml."""
    cfg_root = tmp_path / "cfg"
    cfg_root.mkdir()
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg_root))
    from app.utils import xconfig
    xconfig.reset_cache()

    def write(text: str):
        (cfg_root / "media.yaml").write_text(text, encoding="utf-8")
        xconfig.reset_cache()

    return write


@pytest.fixture(autouse=True)
def _reset_xconfig():
    from app.utils import xconfig
    xconfig.reset_cache()
    yield
    xconfig.reset_cache()


# --------------------------------------------------- tts_engine_chain (pure)

def test_tts_chain_legacy_without_order():
    assert tts_engine_chain("Kokoro", []) == ["Kokoro"]
    assert tts_engine_chain("Kokoro", None) == ["Kokoro"]
    # historical hardwired fallback preserved when no order is configured
    assert tts_engine_chain("ComfyUI", []) == ["ComfyUI", "Kokoro"]


def test_tts_chain_shipped_default_matches_legacy_fallback():
    assert tts_engine_chain("ComfyUI", ["comfyui", "kokoro"]) == ["ComfyUI", "Kokoro"]


def test_tts_chain_gui_choice_keeps_priority():
    chain = tts_engine_chain("Kokoro", ["comfyui", "kokoro", "edge"])
    assert chain == ["Kokoro", "ComfyUI", "Edge TTS"]


def test_tts_chain_selection_not_in_order_goes_first():
    chain = tts_engine_chain("XTTSv2", ["comfyui", "kokoro"])
    assert chain == ["XTTSv2", "ComfyUI", "Kokoro"]


def test_tts_chain_aliases_and_junk_filtered():
    chain = tts_engine_chain(None, ["COMFYUI", "kokoro", "gpt-sovits", "", None, "comfyui"])
    assert chain == ["ComfyUI", "Kokoro"]


def test_tts_chain_single_string_order_tolerated():
    assert tts_engine_chain("Silero", "comfyui") == ["Silero", "ComfyUI"]


def test_tts_chain_none_selection_with_order():
    assert tts_engine_chain(None, ["kokoro", "comfyui"]) == ["Kokoro", "ComfyUI"]


# ------------------------------------------- configured chain via xconfig

def test_configured_chain_reads_media_yaml(media_config):
    media_config("tts_engine_order: [comfyui, kokoro, edge]\n")
    assert configured_tts_chain("ComfyUI") == ["ComfyUI", "Kokoro", "Edge TTS"]


def test_configured_chain_legacy_when_key_absent(media_config):
    media_config("comfyui:\n  enabled: true\n")
    assert configured_tts_chain("ComfyUI") == ["ComfyUI", "Kokoro"]
    assert configured_tts_chain("Silero") == ["Silero"]


def test_configured_chain_seeds_repo_defaults(media_config):
    # media.yaml is seeded from config-defaults/media.yaml on first load
    assert configured_tts_chain("ComfyUI") == ["ComfyUI", "Kokoro"]
    assert configured_tts_chain("Kokoro") == ["Kokoro", "ComfyUI"]


# ------------------------------------------------------ fallback walkers

def test_walk_skips_failures_and_empties():
    seq, hops = [], []

    def attempt(name):
        seq.append(name)
        if name == "ComfyUI":
            raise RuntimeError("down")
        return {"Kokoro": None, "Edge TTS": "edge.wav"}[name]

    def on_fallback(cur, nxt, exc):
        hops.append((cur, nxt, exc))

    res = run_fallback_chain(["ComfyUI", "Kokoro", "Edge TTS"], attempt, on_fallback=on_fallback)

    assert res == "edge.wav"
    assert seq == ["ComfyUI", "Kokoro", "Edge TTS"]
    assert hops[0][:2] == ("ComfyUI", "Kokoro")
    assert isinstance(hops[0][2], RuntimeError)
    assert hops[1] == ("Kokoro", "Edge TTS", None)


def test_walk_first_truthy_short_circuits():
    seq = []

    def attempt(name):
        seq.append(name)
        return "first.wav" if name == "ComfyUI" else "second.wav"

    assert run_fallback_chain(["ComfyUI", "Kokoro"], attempt) == "first.wav"
    assert seq == ["ComfyUI"]


def test_walk_raises_when_last_engine_fails():
    def attempt(name):
        raise RuntimeError(f"{name} boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_fallback_chain(["ComfyUI", "Kokoro"], attempt)

    # legacy single-engine chain: exception propagates as before
    with pytest.raises(RuntimeError):
        run_fallback_chain(["XTTSv2"], attempt)


def test_arun_fallback_walk_same_semantics():
    seq = []

    async def attempt(name):
        seq.append(name)
        if name == "ComfyUI":
            raise RuntimeError("unreachable")
        return {"Kokoro": None, "Edge TTS": "edge.wav"}[name]

    hops = []

    async def go():
        return await arun_fallback_chain(
            ["ComfyUI", "Kokoro", "Edge TTS"],
            attempt,
            on_fallback=lambda c, n, e: hops.append((c, n, e)),
        )

    assert asyncio.run(go()) == "edge.wav"
    assert seq == ["ComfyUI", "Kokoro", "Edge TTS"]
    assert hops == [
        ("ComfyUI", "Kokoro", hops[0][2]),
        ("Kokoro", "Edge TTS", None),
    ]
    assert isinstance(hops[0][2], RuntimeError)


# ------------------------------------------------------- fallback logging

def test_log_tts_engine_fallback_keeps_legacy_wording_and_dedupes(caplog):
    from app.utils import text_to_speech as tts_mod

    owner = types.SimpleNamespace()
    with caplog.at_level(logging.DEBUG, logger="Text-To-Speech Module"):
        tts_mod.log_tts_engine_fallback(owner, "ComfyUI", "Kokoro", None)
        tts_mod.log_tts_engine_fallback(owner, "ComfyUI", "Kokoro", None)  # deduped
        msgs = [(r.levelno, r.getMessage()) for r in caplog.records
                if r.name == "Text-To-Speech Module"]

    assert msgs[0][0] == logging.WARNING
    assert msgs[0][1] == "[Media] ComfyUI workflow produced no audio, falling back to Kokoro"
    assert msgs[1][0] == logging.DEBUG

    owner2 = types.SimpleNamespace()
    tts_mod.log_tts_engine_fallback(owner2, "ComfyUI", "Kokoro", RuntimeError("refused"))
    assert "[Media] ComfyUI unavailable, falling back to Kokoro: refused" in caplog.text


# ------------------------------------- TTSWorker chain runner (no Qt init)

def test_tts_worker_runs_chain_over_fake_engines():
    from app.utils import text_to_speech as tts_mod

    calls = []

    def engine(name):
        async def gen(*args, **kwargs):
            calls.append(name)
            behavior = {"ComfyUI": "raise", "Kokoro": None, "Edge TTS": "edge.wav"}
            if behavior[name] == "raise":
                raise RuntimeError("comfy down")
            return behavior[name]
        return gen

    comfy_handle = types.SimpleNamespace(generate=engine("ComfyUI"))
    worker = types.SimpleNamespace(
        tts_method="ComfyUI",
        character_name="ann", voice_id=None, language="ru",
        xtts=None, edge=None, silero=None, qwen=None, eleven=None,
        kokoro=types.SimpleNamespace(generate_speech_with_kokoro=engine("Kokoro")),
        _get_comfy_tts=lambda: comfy_handle,
        _fallback_warned=set(),
    )
    # dispatch for "Edge TTS" resolves through self.edge; prime it
    worker.edge = types.SimpleNamespace(
        generate_speech_with_edge_tts_sow_system=engine("Edge TTS"))
    # bind the real TTSWorker methods onto the stub
    worker._dispatch_engine = lambda m, l, t: tts_mod.TTSWorker._dispatch_engine(worker, m, l, t)
    worker._run_chain = lambda chain, l, t: tts_mod.TTSWorker._run_chain(worker, chain, l, t)

    loop = asyncio.new_event_loop()
    try:
        out = worker._run_chain(["ComfyUI", "Kokoro", "Edge TTS"], loop, "hi")
    finally:
        loop.close()

    assert out == "edge.wav"
    assert calls == ["ComfyUI", "Kokoro", "Edge TTS"]


def test_tts_pipeline_dispatcher_maps_known_methods():
    from app.utils import text_to_speech as tts_mod

    got = {}

    def engine(tag):
        async def gen(*args, **kwargs):
            got[tag] = args
            return f"{tag}.wav"
        return gen

    worker = types.SimpleNamespace(
        character_name="ann", language="ru", voice_id=None,
        xtts=None, silero=None, qwen=None, eleven=None, comfy_tts=None,
        kokoro=types.SimpleNamespace(generate_speech_with_kokoro=engine("kokoro")),
        edge=types.SimpleNamespace(generate_speech_with_edge_tts_sow_system=engine("edge")),
    )

    assert asyncio.run(PipelinedAdapter._dispatch(worker, "Kokoro", "hey")) == "kokoro.wav"
    assert asyncio.run(PipelinedAdapter._dispatch(worker, "Edge TTS", "ho")) == "edge.wav"
    assert got["kokoro"][0] == "hey"
    assert got["edge"][0] == "ho"


class PipelinedAdapter:
    """Calls PipelinedTTSWorker._dispatch_engine_async against a stub worker."""

    @staticmethod
    async def _dispatch(worker, method, text):
        from app.utils import text_to_speech as tts_mod
        return await tts_mod.PipelinedTTSWorker._dispatch_engine_async(worker, method, text)


# ---------------------------------------- image_provider_chain (pure)

def test_image_chain_legacy_without_order():
    assert image_provider_chain(PROVIDER_A1111, []) == [PROVIDER_A1111]
    assert image_provider_chain(PROVIDER_A1111, None) == [PROVIDER_A1111]
    assert image_provider_chain(PROVIDER_A1111, "") == [PROVIDER_A1111]


def test_image_chain_shipped_default_expands_cloud():
    chain = image_provider_chain(PROVIDER_COMFY_WF, ["comfyui-workflow", "a1111", "cloud"])
    assert chain == [PROVIDER_COMFY_WF, PROVIDER_A1111, PROVIDER_NOVELAI, PROVIDER_DALLE, PROVIDER_FLUX]


def test_image_chain_selection_keeps_priority():
    chain = image_provider_chain(PROVIDER_NOVELAI, ["comfyui-workflow", "novelai"])
    assert chain == [PROVIDER_NOVELAI, PROVIDER_COMFY_WF]


def test_image_chain_aliases_dedupe_unknown_dropped():
    chain = image_provider_chain(None, ["A1111", "automatic1111", "comfy", "midjourney", ""])
    assert chain == [PROVIDER_A1111, PROVIDER_COMFY_WF]


# ------------------------------- ImageGenerator integration (stub config)

class _StubSettings:
    def __init__(self, values):
        self.values = values

    def get_main_setting(self, key):
        return self.values.get(key)


class _StubAPI:
    def __init__(self, tokens):
        self.tokens = tokens

    def get_token(self, key):
        return self.tokens.get(key)


def _make_generator(monkeypatch, provider_value, order, calls, outcomes):
    from app.utils import image_generator as ig

    gen = ig.ImageGenerator.__new__(ig.ImageGenerator)
    gen.config = _StubSettings({
        "image_provider": provider_value,
        "image_prefix_prompt": None,
        "image_negative_prompt": None,
        "image_width": 512, "image_height": 768, "image_steps": 20,
    })
    gen.api_config = _StubAPI({})

    if order is not None:
        monkeypatch.setattr(ig, "_read_provider_order", lambda: order)

    def runner(name, meth):
        async def fn(*args, **kwargs):
            calls.append(name)
            result = outcomes.get(name)
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(gen, meth, fn)
        return fn

    runner("comfy", "generate_comfy_workflow")
    runner("a1111", "generate_a1111")
    runner("novelai", "generate_novelai")
    runner("dalle", "generate_dalle")
    runner("flux", "generate_flux")

    async def save(data, character_name):
        calls.append(("save", data))
        return "saved.png" if data else None
    monkeypatch.setattr(gen, "save_image", save)

    return gen


def test_image_generation_falls_through_failed_provider(monkeypatch):
    from app.utils import image_generator as ig

    calls = []
    gen = _make_generator(
        monkeypatch, "Automatic1111", ["comfyui-workflow", "a1111"], calls,
        {"a1111": None, "comfy": b"PNGDATA"},
    )

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out == "saved.png"
    assert calls[0] == "a1111" and calls[1] == "comfy"
    assert calls[2] == ("save", b"PNGDATA")


def test_image_generation_success_never_tries_rest(monkeypatch):
    from app.utils import image_generator as ig

    calls = []
    gen = _make_generator(
        monkeypatch, "ComfyUI Workflow", ["comfyui-workflow", "a1111", "cloud"], calls,
        {"comfy": b"PNGDATA"},
    )

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out == "saved.png"
    assert calls == ["comfy", ("save", b"PNGDATA")]


def test_image_generation_empty_order_is_legacy_single_shot(monkeypatch):
    from app.utils import image_generator as ig

    calls = []
    gen = _make_generator(
        monkeypatch, "Automatic1111", [], calls, {"a1111": b"AAA", "comfy": b"WASTED"},
    )

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out == "saved.png"
    assert calls == ["a1111", ("save", b"AAA")]
    assert "comfy" not in calls


def test_image_generation_all_fail_yields_none(monkeypatch):
    from app.utils import image_generator as ig

    calls = []
    gen = _make_generator(
        monkeypatch, "ComfyUI Workflow", ["comfyui-workflow", "cloud"], calls,
        {"comfy": None, "novelai": None, "dalle": None, "flux": None},
    )

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out is None
    assert calls[:-1] == ["comfy", "novelai", "dalle", "flux"]
    assert calls[-1] == ("save", None)


def test_image_generation_exception_advances_except_on_last(monkeypatch):
    from app.utils import image_generator as ig

    calls = []
    gen = _make_generator(
        monkeypatch, "ComfyUI Workflow", ["comfyui-workflow", "a1111"], calls,
        {"comfy": RuntimeError("wf broken"), "a1111": None},
    )

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out is None
    assert calls == ["comfy", "a1111", ("save", None)]


def test_image_generation_reads_media_yaml_order(monkeypatch, media_config):
    media_config("image_provider_order: [a1111, comfyui-workflow]\n")

    calls = []
    gen = _make_generator(monkeypatch, "Automatic1111", None, calls, {"a1111": None, "comfy": b"PNG"})

    out = asyncio.run(gen.generate_image("prompt", "ann"))

    assert out == "saved.png"
    assert calls == ["a1111", "comfy", ("save", b"PNG")]
