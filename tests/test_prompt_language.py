"""Reply-language directive in the system prompt (config ui.yaml `reply_language`).

Covers:
  - auto (default from config-defaults/ui.yaml) -> "same language" directive
  - explicit code (ru, de, ...) -> "Always respond in <Language>"
  - key missing (legacy ui.yaml) -> no directive line at all
  - wiring: directive really lands in the messages passed to the provider
"""

import os
from pathlib import Path

import pytest

from app.utils import xconfig


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SOW_CONFIG_DIR", str(tmp_path / "cfg"))
    xconfig.reset_cache()
    yield tmp_path / "cfg"


@pytest.fixture
def engine(monkeypatch):
    from app.utils.ai_clients.prompt_engine import PromptEngine

    eng = PromptEngine()
    # Keep prompt assembly hermetic: no soul-memory file IO / embedder.
    monkeypatch.setattr(eng, "is_soul_memory_enabled", lambda: False)
    return eng


def _set_ui_yaml(config_root, text):
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "ui.yaml").write_text(text, encoding="utf-8")
    xconfig.reset_cache()


def _system_content(engine):
    messages, _ = engine.build_system_prompt_blocks(
        character_name="Test Char",
        user_name="User",
        user_description="a tester",
        chat_messages=[{"role": "user", "content": "Привет, как дела?"}],
        user_message="Привет, как дела?",
    )
    assert messages, "prompt engine must return at least one message"
    assert messages[0]["role"] == "system"
    return messages[0]["content"]


# --- pure mapping -----------------------------------------------------------


def test_auto_directive_mentions_same_language():
    from app.utils.ai_clients.prompt_engine import get_language_directive

    directive = get_language_directive("auto")
    assert directive is not None
    assert "same language" in directive


def test_ru_directive_names_russian():
    from app.utils.ai_clients.prompt_engine import get_language_directive

    directive = get_language_directive("ru")
    assert directive is not None
    assert "Russian" in directive


def test_other_codes_and_unknown():
    from app.utils.ai_clients.prompt_engine import get_language_directive

    assert "German" in get_language_directive("de")
    # Unknown code passes through verbatim as the language name.
    assert "Always respond in xx" in get_language_directive("xx")


def test_missing_or_off_values_yield_no_directive():
    from app.utils.ai_clients.prompt_engine import get_language_directive

    for value in (None, "", "none", "off"):
        assert get_language_directive(value) is None


# --- wiring through build_system_prompt_blocks -------------------------------


def test_default_seeded_config_uses_auto(config_root, engine):
    # Fresh config root: ui.yaml is seeded from config-defaults/ui.yaml.
    cfg = xconfig.load("ui")
    assert cfg.get("reply_language") == "auto"

    content = _system_content(engine)
    assert "same language" in content


def test_explicit_ru_reaches_system_prompt(config_root, engine):
    _set_ui_yaml(config_root, "reply_language: ru\n")
    content = _system_content(engine)
    assert "Always respond in Russian" in content
    assert "same language" not in content


def test_missing_key_keeps_prompt_without_directive(config_root, engine):
    # Legacy ui.yaml (no reply_language key) must not gain any language line.
    _set_ui_yaml(config_root, "theme: dark\n")
    content = _system_content(engine)
    assert "Always respond" not in content
    assert "same language" not in content
    # Base preset prompt is still intact.
    assert "neverending story" in content
