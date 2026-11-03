"""Faster-Whisper STT with local-weights-first configuration."""

import logging

from .base import BaseSTT
from app.utils import xconfig

logger = logging.getLogger("WhisperProvider")


class FasterWhisperSTT(BaseSTT):
    def __init__(self, model_path=None, device=None, compute_type=None,
                 model_cls=None):
        cfg = xconfig.load("llm").get("stt", {}) or {}
        self.model_path = model_path or cfg.get("whisper_model_path") or ""
        self.device = device or cfg.get("device") or "auto"
        self.compute_type = compute_type or cfg.get("compute_type") or "float16"
        self._model_cls = model_cls
        self._model = None

    @property
    def _target(self) -> str:
        # Пустой путь в конфиге ⇒ имя модели из библиотеки (может скачать);
        # заполненный ⇒ локальные веса, без сети.
        return self.model_path if self.model_path else "small"

    def _load(self):
        if self._model is not None:
            return
        cls = self._model_cls
        if cls is None:
            from faster_whisper import WhisperModel as _W
            cls = _W
        logger.info(
            "[STT] loading whisper weights: %s (device=%s, compute=%s)",
            self._target, self.device, self.compute_type,
        )
        self._model = cls(
            model_size_or_path=self._target,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, wav_path, language=None) -> str:
        self._load()
        segments, _info = self._model.transcribe(str(wav_path), language=language)
        return "".join(s.text for s in segments).strip()
