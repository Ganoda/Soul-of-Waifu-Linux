"""Async ComfyUI API client.

Cycle: POST /prompt → poll GET /history/{id} → GET /view → local cache.
All connection/timeout/HTTP>=500 failures surface as ComfyUIUnavailable;
executed-but-empty workflows return an empty result dict instead.
"""

import asyncio
import logging
import mimetypes
import time
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlencode

import aiohttp

from app.utils import xconfig

logger = logging.getLogger("ComfyUIClient")


class ComfyUIUnavailable(Exception):
    pass


def _cfg():
    c = xconfig.load("media")
    return {
        "base_url": c.get("comfyui.base_url", "http://127.0.0.1:8188"),
        "enabled": bool(c.get("comfyui.enabled", True)),
        "timeout_s": int(c.get("comfyui.timeout_s", 300)),
        "poll_interval_s": float(c.get("comfyui.poll_interval_s", 0.5)),
        "cache_dir": str(c.get("comfyui.cache_dir", "app/cache/comfy")),
    }


class ComfyUIClient:
    def __init__(self, base_url=None, timeout_s=None, poll_interval_s=None,
                 cache_dir=None):
        cfg = _cfg()
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.enabled = bool(cfg.get("enabled", True))
        self.timeout_s = int(timeout_s if timeout_s is not None else cfg["timeout_s"])
        self.poll_interval_s = float(
            poll_interval_s if poll_interval_s is not None else cfg["poll_interval_s"]
        )
        self.cache_dir = Path(cache_dir or cfg["cache_dir"])
        self._session: aiohttp.ClientSession | None = None
        self._healthy: bool | None = None

    @property
    def healthy(self) -> bool:
        return bool(self._healthy)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        return self._session

    async def ping(self) -> bool:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/system_stats",
                                   timeout=aiohttp.ClientTimeout(total=2)) as resp:
                self._healthy = resp.status < 400
                return self._healthy
        except Exception as e:
            logger.debug("[ComfyUI] ping failed: %s", e)
            self._healthy = False
            return False

    async def run(self, workflow: Mapping, *, node_inputs: Mapping[str, Mapping] = {},
                  prefix: str = "job", cache_dir=None) -> dict:
        out_dir = Path(cache_dir) if cache_dir else self.cache_dir

        if not self.enabled:
            raise ComfyUIUnavailable("ComfyUI disabled in media.yaml (comfyui.enabled: false)")

        if self._healthy is not True and not (await self.ping()):
            raise ComfyUIUnavailable(f"ComfyUI unreachable at {self.base_url}")

        try:
            patched = self._apply_inputs(workflow, node_inputs)
            prompt_id = await self._submit(patched)
            outputs = await self._wait_outputs(prompt_id)
        except ComfyUIUnavailable:
            raise
        except aiohttp.ClientError as e:
            self._healthy = False
            raise ComfyUIUnavailable(str(e)) from e
        except asyncio.TimeoutError as e:
            self._healthy = False
            raise ComfyUIUnavailable(f"timeout: {e}") from e

        return await self._fetch_artifacts(prompt_id, outputs, prefix, out_dir)

    @staticmethod
    def _apply_inputs(workflow: Mapping, node_inputs: Mapping) -> dict:
        patched = json_deepcopy(workflow)
        for node_id, fields in node_inputs.items():
            bucket = patched.setdefault(str(node_id), {})
            inputs = bucket.setdefault("inputs", {})
            inputs.update(fields)
        return patched

    async def _submit(self, patched: dict) -> str:
        session = await self._get_session()
        async with session.post(f"{self.base_url}/prompt", json={"prompt": patched}) as resp:
            if resp.status >= 500:
                raise aiohttp.ClientResponseError(resp.request_info, resp.history,
                                                  status=resp.status)
            body = await resp.json(content_type=None)
        pid = body.get("prompt_id")
        if not pid:
            raise ComfyUIUnavailable(f"/prompt returned no prompt_id: {body}")
        return pid

    async def _wait_outputs(self, prompt_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while True:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/history/{prompt_id}") as resp:
                data = await resp.json(content_type=None)
            entry = data.get(prompt_id) or {}
            outputs = entry.get("outputs")
            if outputs is not None:
                return outputs
            if time.monotonic() > deadline:
                raise ComfyUIUnavailable(f"history wait timed out for {prompt_id}")
            await asyncio.sleep(self.poll_interval_s)

    async def _fetch_artifacts(self, prompt_id: str, outputs: dict,
                               prefix: str, out_dir: Path) -> dict:
        result = {"audio": [], "images": [], "video": []}
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        stamp = int(time.time())
        session = await self._get_session()

        for _node, artifacts in (outputs or {}).items():
            if not isinstance(artifacts, dict):
                continue
            for key, items in artifacts.items():
                kind = ("audio" if key == "audios"
                        else "images" if key == "images"
                        else "video" if key == "videos" else None)
                if kind is None:
                    continue
                for item in items:
                    fname = item.get("filename", "")
                    params = {"filename": fname,
                              "subfolder": item.get("subfolder", ""),
                              "type": item.get("type", "output")}
                    url = f"{self.base_url}/view?{urlencode(params)}"
                    try:
                        async with session.get(url) as resp:
                            body = await resp.read()
                            ctype = resp.headers.get("Content-Type", "")
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning("[ComfyUI] /view fetch failed (%s): %s", fname, e)
                        continue

                    ext = Path(fname).suffix or _ext_from_mime(ctype)
                    path = out_dir / f"{prefix}_{stamp}_{n}{ext}"
                    n += 1
                    path.write_bytes(body)
                    result[kind].append(path.resolve())
        return result


def json_deepcopy(obj):
    import copy
    if isinstance(obj, Mapping):
        return {k: json_deepcopy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_deepcopy(v) for v in obj]
    return obj


def _ext_from_mime(ctype: str) -> str:
    guess = mimetypes.guess_extension((ctype or "").split(";")[0].strip()) or ".bin"
    return guess if guess != ".jpe" else ".jpg"
