"""Kokoro fallback adapter delegating to the legacy SoW TTS class.

Heavy model load happens inside the legacy class on first generate; the
adapter wraps a lazily-constructed singleton getter so importing this
module stays cheap and headless-testable.
"""

import asyncio
import logging

from .base import BaseTTS

logger = logging.getLogger("KokoroProvider")

_singleton = None


def _legacy_kokoro():
    global _singleton
    if _singleton is None:
        from app.utils.text_to_speech import KokoroTTS_SOW_System
        _singleton = KokoroTTS_SOW_System()
    return _singleton


class KokoroAdapter(BaseTTS):
    def __init__(self, tts_singleton_getter=None):
        self._getter = tts_singleton_getter or _legacy_kokoro

    async def generate(self, text: str, character_name: str):
        kokoro = self._getter()
        return await asyncio.to_thread(
            lambda: asyncio.run(
                kokoro.generate_speech_with_kokoro(text, character_name)
            )
        )


def register_defaults() -> None:
    from . import registry

    def make_llm(base_url="", api_key="", model=""):
        from .openai_compat_llm import OpenAICompatLLM
        return OpenAICompatLLM(base_url=base_url, api_key=api_key, model=model)

    def llm_factory(**kw):
        if not kw.get("base_url"):
            from app.utils import xconfig
            c = xconfig.load("llm")
            kw.setdefault("base_url", c.get("backend", ""))
            kw.setdefault("api_key", c.get("api_key", "") or "")
            kw.setdefault("model", c.get("model", ""))
        return make_llm(**{k: v for k, v in kw.items() if k in ("base_url", "api_key", "model")})

    def comfy_tts_factory(**kw):
        accepted = {k: v for k, v in kw.items()
                    if k in ("client", "workflow_spec") and v is not None}
        if not kw.get("client"):
            accepted.pop("client", None)
            return ComfyUITTS(accepted.get("workflow_spec"))
        return ComfyUITTS(kw.get("client"), accepted.get("workflow_spec"))

    def comfy_image_factory(**kw):
        if not kw.get("client"):
            from .comfy_tts import ComfyUIImageGen  # same module family; avoids dup import at top
            return ComfyUIImageGen(None, kw.get("workflow_spec"))
        return ComfyUIImageGen(kw.get("client"), kw.get("workflow_spec"))

    def kokoro_factory(getter=None):
        return KokoroAdapter(getter)

    def whisper_factory(model_cls=None):
        kwargs = {}
        if model_cls is not None:
            kwargs["model_cls"] = model_cls
        return FasterWhisperSTT(**kwargs)

    registry.register("llm", "openai-compat", llm_factory)
    registry.register("tts", "comfyui", comfy_tts_factory)
    registry.register("tts", "kokoro", kokoro_factory)
    registry.register("stt", "faster-whisper", whisper_factory)
    registry.register("image", "comfyui", comfy_image_factory)
