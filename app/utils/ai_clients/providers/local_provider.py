import logging
import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError
from app.utils.ai_clients.base_provider import BaseAIProvider
from app.utils.ai_clients.local_server_manager import (
    ensure_embedded_llm_server,
    recent_embedded_server_error,
)

logger = logging.getLogger("Local Provider")

REASONING_EFFORT_TO_BUDGET = {
    "low": 1000,
    "medium": 4000,
    "high": 10000,
    "xhigh": 20000,
}

class LocalProvider(BaseAIProvider):
    # Streaming requests may carry tool schemas: generate_stream then also
    # emits {"tool_call_deltas": [...]} events alongside text chunks, letting
    # the GUI run tools inline instead of a separate non-streaming pre-pass.
    supports_stream_tools = True

    def __init__(self, port: int = 48596, advanced_params: dict = None,
                 base_url: str = None, api_key: str = None,
                 model: str = "local-model", sampling_defaults: dict = None):
        if base_url:
            # Canonical OpenAI-compat URL is {base}/v1/chat/completions;
            # the /v1 suffix is appended the same way the provider layer's
            # OpenAICompatLLM does, so the user may enter the address either way.
            base = base_url.rstrip("/")
            if not base.endswith("/v1"):
                base = f"{base}/v1"
            self.base_url = base
        else:
            self.base_url = f"http://127.0.0.1:{port}/v1"
        self.api_key = api_key or "no-key-required"
        self.model = model
        self.advanced_params = advanced_params or {}
        self.sampling_defaults = sampling_defaults or {}

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)
            )
        )

    def _param(self, kwargs: dict, key: str, fallback):
        """Request parameter: caller wins, then llm.yaml default, then builtin."""
        if key in kwargs:
            return kwargs[key]
        return self.sampling_defaults.get(key, fallback)

    def _connection_error_message(self, e: Exception) -> str:
        logger.error(f"Local API Connection Error: {e}")
        message = (
            "\n⚠️ Local API Error: Could not reach the local server. "
            "It may still be loading the model, may have crashed, or the port may be blocked."
        )
        # If the embedded llama-server just failed to start, explain WHY it
        # is offline instead of leaving the user with a bare connection error.
        root_cause = recent_embedded_server_error()
        if root_cause:
            message += f"\n↳ {root_cause}"
        return message

    def _apply_thinking_budget(self, extra_body: dict, kwargs: dict, payload: dict) -> None:
        reasoning_mode = kwargs.get("reasoning_mode")
        if reasoning_mode is False:
            extra_body["thinking_budget_tokens"] = 0
            return

        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            budget = REASONING_EFFORT_TO_BUDGET.get(reasoning_effort, 4000)
            extra_body["thinking_budget_tokens"] = budget
            payload["max_tokens"] = max(payload["max_tokens"], budget + 1024)

    async def generate_stream(self, messages: list[dict], **kwargs):
        # Embedded llama-server must be up before the first request; chat,
        # regenerate and NPC paths reach us without any GUI-side bootstrap.
        try:
            await ensure_embedded_llm_server()
        except Exception as e:
            logger.error(f"Embedded LLM server bootstrap failed: {e}")
            # Fall through: the request below surfaces the recorded root cause.

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": self._param(kwargs, "temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }

        tools = kwargs.get("tools")
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        stop_sequences = self._param(kwargs, "stop", None)
        if stop_sequences:
            payload["stop"] = stop_sequences

        extra_body = dict(self.advanced_params) if self.advanced_params else {}
        repeat_penalty = self._param(kwargs, "repeat_penalty", None)
        if repeat_penalty is not None:
            extra_body["repeat_penalty"] = repeat_penalty
        self._apply_thinking_budget(extra_body, kwargs, payload)
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            completion = await self.client.chat.completions.create(**payload)
            async for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # Tool-call fragments surface as non-string events; the GUI
                # stream wrapper accumulates and executes them inline.
                if getattr(delta, "tool_calls", None):
                    yield {"tool_call_deltas": delta.tool_calls}
                if delta.content:
                    yield delta.content
        except (APIConnectionError, APITimeoutError) as e:
            yield self._connection_error_message(e)
        except Exception as e:
            logger.error(f"Local API Stream Error: {e}")
            yield f"\n⚠️ Local API Error: {str(e)}"

    async def generate_summary(self, messages: list[dict], **kwargs):
        try:
            await ensure_embedded_llm_server()
        except Exception as e:
            logger.error(f"Embedded LLM server bootstrap failed (summary): {e}")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": self._param(kwargs, "temperature", 0.5),
            "top_p": kwargs.get("top_p", 0.9),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.8),
            "presence_penalty": kwargs.get("presence_penalty", 0.3)
        }

        stop_sequences = self._param(kwargs, "stop", None)
        if stop_sequences:
            payload["stop"] = stop_sequences

        reasoning_mode = kwargs.get("reasoning_mode", False)
        extra_body = {}
        if reasoning_mode is False:
            extra_body["thinking_budget_tokens"] = 0
        repeat_penalty = self._param(kwargs, "repeat_penalty", None)
        if repeat_penalty is not None:
            extra_body["repeat_penalty"] = repeat_penalty
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            completion = await self.client.chat.completions.create(**payload)
            async for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Local API Connection Error (summary): {e}")
            yield ""
        except Exception as e:
            logger.error(f"Local API Summary Error: {e}")
            yield ""

    async def generate(self, messages: list[dict], tools: list = None, **kwargs) -> dict:
        try:
            await ensure_embedded_llm_server()
        except Exception as e:
            logger.error(f"Embedded LLM server bootstrap failed (generate): {e}")

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": self._param(kwargs, "temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        stop_sequences = self._param(kwargs, "stop", None)
        if stop_sequences:
            payload["stop"] = stop_sequences

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        extra_body = dict(self.advanced_params) if self.advanced_params else {}
        repeat_penalty = self._param(kwargs, "repeat_penalty", None)
        if repeat_penalty is not None:
            extra_body["repeat_penalty"] = repeat_penalty
        reasoning_mode = kwargs.get("reasoning_mode")
        if reasoning_mode is False:
            extra_body["thinking_budget_tokens"] = 0
        if extra_body:
            payload["extra_body"] = extra_body

        try:
            completion = await self.client.chat.completions.create(**payload)
            msg = completion.choices[0].message
            return {
                "content": msg.content,
                "tool_calls": msg.tool_calls
            }
        except (APIConnectionError, APITimeoutError) as e:
            logger.error(f"Local API Connection Error (generate): {e}")
            return {"content": None, "tool_calls": None}
        except Exception as e:
            logger.error(f"Local API Generate Error: {e}")
            return {"content": None, "tool_calls": None}
