"""Abstract provider interfaces for the Linux provider layer.

All IO-bound methods are async-first; blocking implementations are
expected to run their heavy work in threads via asyncio.to_thread at the
caller side or internally.
"""

import abc

KIND_LLM = "llm"
KIND_TTS = "tts"
KIND_STT = "stt"
KIND_IMAGE = "image"


class BaseLLM(abc.ABC):
    @abc.abstractmethod
    async def chat(self, messages: list, *, stream: bool = False, **params):
        """Send a chat completion; returns str (or AsyncIterator[str] when stream)."""


class BaseTTS(abc.ABC):
    @abc.abstractmethod
    async def generate(self, text: str, character_name: str):
        """Synthesize speech; returns absolute file path or None."""


class BaseSTT(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, wav_path, language=None) -> str:
        """Blocking transcription; callers must invoke off the Qt loop."""


class BaseImageGen(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str, negative: str = "", **kw):
        """Generate an image; returns image bytes or None."""
