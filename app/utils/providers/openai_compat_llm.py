"""OpenAI-compatible chat-completions client for local llama.cpp servers."""

import logging

import aiohttp

logger = logging.getLogger("LLMProvider")


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str = "", model: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def chat(self, messages: list, *, stream: bool = False, **params) -> str:
        session = await self._get_session()
        payload = {"model": self.model, "messages": messages, "stream": stream}
        payload.update(params)
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            raise ConnectionError(f"LLM backend unreachable at {url}: {e}") from e

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = (choices[0] or {}).get("message") or {}
        return message.get("content", "")
