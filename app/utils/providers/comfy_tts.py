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


async def probe_comfyui_health(client) -> bool:
    try:
        return await client.ping()
    except Exception:
        return False
