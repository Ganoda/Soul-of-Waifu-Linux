import pytest

from app.utils.providers.base import (
    KIND_IMAGE,
    KIND_LLM,
    KIND_STT,
    KIND_TTS,
    BaseImageGen,
    BaseLLM,
    BaseSTT,
    BaseTTS,
)
from app.utils.providers import registry


def test_kind_constants():
    assert (KIND_LLM, KIND_TTS, KIND_STT, KIND_IMAGE) == ("llm", "tts", "stt", "image")


def test_register_and_create_roundtrip_with_kwargs():
    registry.register("tts", "dummy", lambda **kw: {"got": kw})
    obj = registry.create("tts", "dummy", speed=2)
    assert obj == {"got": {"speed": 2}}


def test_duplicate_registration_replaces_silently():
    registry.register("llm", "dup", lambda: "first")
    registry.register("llm", "dup", lambda: "second")
    assert registry.create("llm", "dup") == "second"


def test_available_sorted_and_scoped_by_kind():
    registry.register("stt", "zeta", lambda: None)
    registry.register("stt", "alpha", lambda: None)
    assert registry.available("stt") == ["alpha", "zeta"]
    assert registry.available("image") == []


def test_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        registry.create("llm", "nope")


def test_abstract_base_cannot_instantiate_without_implementations():
    class Incomplete(BaseTTS):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_concrete_subclass_instantiates():
    class Ok(BaseLLM):
        async def chat(self, messages, *, stream=False, **params):
            return "ok"

    assert isinstance(Ok(), BaseLLM)
