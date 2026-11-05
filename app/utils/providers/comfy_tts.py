"""ComfyUI-backed TTS/ImageGen providers with drop-in workflow discovery."""

import asyncio
import logging
from pathlib import Path

from .base import BaseImageGen, BaseTTS
from app.utils.media import workflow_loader

logger = logging.getLogger("ComfyMediaProviders")


class ComfyUITTS(BaseTTS):
    def __init__(self, client, workflow_spec=None):
        self.client = client
        self.spec = workflow_spec

    async def generate(self, text: str, character_name: str):
        from app.utils.media.comfy_client import ComfyUIUnavailable

        if self.client is not None and not (await self.client.ping()):
            raise ComfyUIUnavailable(f"ComfyUI unreachable at {self.client.base_url}")

        if self.spec is None:
            specs = workflow_loader.scan("tts")
            if not specs:
                raise RuntimeError("no tts workflow found in drop-in dir")
            self.spec = specs[0]
        if self.client is None:
            from app.utils.media.comfy_client import ComfyUIClient
            self.client = ComfyUIClient()

        field = self.spec.text_input_field or "text"
        node = self.spec.text_node_id
        result = await self.client.run(
            self.spec.raw,
            node_inputs={node: {field: text}},
            prefix=f"tts_{character_name}",
        )
        audio = result.get("audio") or []
        return str(audio[0]) if audio else None


class ComfyUIImageGen(BaseImageGen):
    def __init__(self, client, workflow_spec=None):
        self.client = client
        self.spec = workflow_spec

    async def generate(self, prompt: str, negative: str = "", **kw):
        if self.spec is None:
            specs = workflow_loader.scan("image")
            if not specs:
                raise RuntimeError("no image workflow found in drop-in dir")
            self.spec = specs[0]
        if self.client is None:
            from app.utils.media.comfy_client import ComfyUIClient
            self.client = ComfyUIClient()

        field = self.spec.text_input_field or "text"
        inputs = {self.spec.text_node_id: {field: prompt}}
        neg_id = _find_negative_node(self.spec.raw)
        if neg_id and negative:
            inputs[neg_id] = {"text": negative}

        result = await self.client.run(
            self.spec.raw, node_inputs=inputs, prefix="img"
        )
        images = result.get("images") or []
        if not images:
            return None
        # caller wants bytes per BaseImageGen contract — read first artifact
        data: bytes = Path(images[0]).read_bytes()
        try:
            Path(images[0]).unlink()
        except OSError:
            pass
        return data


def _find_negative_node(raw: dict):
    for node_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type", "")).lower()
        title = ""
        meta = node.get("_meta") or {}
        if isinstance(meta, dict) and isinstance(meta.get("title"), str):
            title = meta["title"].lower()
        if "negativ" in cls or "negativ" in title:
            return node_id
    return None


def pick_tts_engine(order: list) -> list:
    """Trim an ordered engine preference list to actually-registered engines."""
    registered = set(__import__("app.utils.providers.registry", fromlist=["available"]).available("tts"))
    out = []
    for name in order:
        if name in registered and name not in out:
            out.append(name)
    return out


# --- Phase-B: media.yaml tts_engine_order -----------------------------------

#: media.yaml shorthand -> the TTS method string used by the GUI/dispatcher.
TTS_ENGINE_ALIASES = {
    "comfyui": "ComfyUI",
    "kokoro": "Kokoro",
    "xttsv2": "XTTSv2",
    "xtts": "XTTSv2",
    "edge": "Edge TTS",
    "edge tts": "Edge TTS",
    "edge-tts": "Edge TTS",
    "silero": "Silero",
    "qwen3": "Qwen-3 TTS",
    "qwen-3 tts": "Qwen-3 TTS",
    "qwen-3-tts": "Qwen-3 TTS",
    "elevenlabs": "ElevenLabs",
    "eleven labs": "ElevenLabs",
}


def normalize_engine_list(order) -> list:
    """Coerce a configured order value to a list of clean lowercase names."""
    if isinstance(order, str):
        order = [order]
    if not isinstance(order, (list, tuple)):
        return []
    out = []
    for name in order:
        cleaned = str(name or "").strip().lower()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def tts_engine_chain(selected_method, order) -> list:
    """Build the TTS fallback chain for one utterance.

    Semantics (Phase-B):
    - ``order`` (media.yaml ``tts_engine_order``) is an iterable of engine
      names in priority order; unknown names are dropped, duplicates collapse;
    - the GUI-selected engine keeps priority: it is always first when set,
      and the remaining ``order`` entries follow as *extra* fallback engines;
    - an empty/missing ``order`` preserves legacy behaviour exactly:
      only the selected engine runs, plus Kokoro behind ComfyUI (the
      historical hardwired fallback).
    """
    canon = [TTS_ENGINE_ALIASES[n] for n in normalize_engine_list(order)
             if n in TTS_ENGINE_ALIASES]
    raw_sel = str(selected_method or "").strip()
    sel = TTS_ENGINE_ALIASES.get(raw_sel.lower(), raw_sel or None)

    if not canon:
        if sel == "ComfyUI":
            return [sel, "Kokoro"]
        return [sel] if sel else []

    chain = [m for m in canon if m != sel]
    if sel:
        chain.insert(0, sel)
    return chain


def configured_tts_chain(selected_method) -> list:
    """tts_engine_chain() with the order taken from media.yaml."""
    from app.utils import xconfig
    order = xconfig.load("media").get("tts_engine_order")
    return tts_engine_chain(selected_method, order)


def run_fallback_chain(chain, attempt, on_fallback=None):
    """Walk an engine-priority chain synchronously.

    ``attempt(name)`` runs one engine; a truthy result wins immediately.
    A failing engine (exception or falsy result) moves to the next entry;
    the *last* engine keeps legacy semantics — its exception propagates and
    its falsy result is returned as-is. ``on_fallback(current, next, exc)``
    is invoked for each non-final failure (may be None).
    """
    result = None
    for idx, name in enumerate(chain):
        last = idx == len(chain) - 1
        try:
            result = attempt(name)
        except Exception as e:
            if last:
                raise
            if on_fallback:
                on_fallback(name, chain[idx + 1], e)
            continue
        if result:
            return result
        if not last:
            if on_fallback:
                on_fallback(name, chain[idx + 1], None)
    return result


async def arun_fallback_chain(chain, attempt, on_fallback=None):
    """Async twin of :func:`run_fallback_chain` (``attempt`` is awaited)."""
    result = None
    for idx, name in enumerate(chain):
        last = idx == len(chain) - 1
        try:
            result = await attempt(name)
        except Exception as e:
            if last:
                raise
            if on_fallback:
                on_fallback(name, chain[idx + 1], e)
            continue
        if result:
            return result
        if not last:
            if on_fallback:
                on_fallback(name, chain[idx + 1], None)
    return result


async def probe_comfyui_health(client) -> bool:
    try:
        return await client.ping()
    except Exception:
        return False
