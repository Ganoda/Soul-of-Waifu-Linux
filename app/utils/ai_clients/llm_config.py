"""Single reader for the modular llm.yaml LLM section.

Both the factory (which provider instance to build) and the local server
manager (whether to spawn an embedded llama-server at all) need the same
answer: is an external OpenAI-compatible backend configured in llm.yaml?
Keeping it here avoids the two call sites drifting apart.
"""

import logging

logger = logging.getLogger("LLM Config")


def external_backend() -> dict | None:
    """Return llm.yaml connection params, or None when no backend is set.

    A falsy/absent "backend" key means "launch the embedded llama-server
    like before", while any non-empty value (llama-server, vLLM, Ollama -
    any OpenAI-compatible server) is treated as the user-managed endpoint.
    Sampling values from llm.yaml act as request defaults only; explicit
    per-call kwargs keep winning over them.
    """
    from app.utils import xconfig

    c = xconfig.load("llm")
    base_url = (c.get("backend") or "").strip()
    if not base_url:
        return None

    sampling_defaults = {}
    temperature = c.get("temperature")
    if temperature is not None:
        sampling_defaults["temperature"] = temperature
    repeat_penalty = c.get("repeat_penalty")
    if repeat_penalty is not None:
        sampling_defaults["repeat_penalty"] = repeat_penalty
    stop_tokens = c.get("stop_tokens")
    if stop_tokens:
        sampling_defaults["stop"] = stop_tokens

    return {
        "base_url": base_url,
        "api_key": (c.get("api_key") or "").strip(),
        "model": (c.get("model") or "").strip() or "local-model",
        "sampling_defaults": sampling_defaults,
    }
