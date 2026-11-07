"""mmproj (multimodal projector) support for the embedded llama-server launch.

Covers sibling auto-detection next to the selected GGUF, the llm.yaml
mmproj_path override, and the "--mmproj" flag wiring in the built command.
"""

import asyncio
import logging

import pytest

from app.utils import xconfig
from app.utils.ai_clients import local_server_manager as lsm
from app.utils.ai_clients.local_server_manager import find_mmproj_sibling
from app.utils.ai_clients.local_server_manager import LocalServerManager


GEMMA_MODEL = "Huihui-gemma-4-12B-it-abliterated-GGUF.gguf"
GEMMA_PROJECTOR = "Huihui-gemma-4-12B-it-abliterated-projector.gguf"
GEMMA_EXTRA_QUANT = "Huihui-gemma-4-12B-it-abliterated-Q8_0.gguf"
QWEN_MODEL = "Swift-Qwen3.8-Flash-Next-Q4_K_M.gguf"
QWEN_MMPROJ = "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg))
    xconfig.reset_cache()
    yield cfg


@pytest.fixture
def module_root(tmp_path, monkeypatch):
    monkeypatch.setattr(lsm, "MODULE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def manager(module_root):
    return LocalServerManager(ui=None)


def _touch(directory, *names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"\x00fake-gguf")


# --- sibling auto-detection -------------------------------------------------


def test_detects_pairing_projector_next_to_model(tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR, GEMMA_EXTRA_QUANT)

    assert find_mmproj_sibling(models / GEMMA_MODEL) == str(models / GEMMA_PROJECTOR)


def test_detects_llamacpp_style_mmproj_prefix(tmp_path):
    models = tmp_path / "models"
    _touch(models, QWEN_MODEL, QWEN_MMPROJ)

    assert find_mmproj_sibling(models / QWEN_MODEL) == str(models / QWEN_MMPROJ)


def test_mmproj_prefix_matches_model_without_shared_leading_token(tmp_path):
    # mmproj-<Model>-<quant>.gguf vs <Model>-<quant>.gguf: the in-order token
    # fallback pairs them even though "Swift" is absent from the model name.
    models = tmp_path / "models"
    plain_model = "Qwen3.8-Flash-Next-Q4_K_M.gguf"
    _touch(models, plain_model, QWEN_MMPROJ)

    assert find_mmproj_sibling(models / plain_model) == str(models / QWEN_MMPROJ)


def test_foreign_family_mmproj_is_ignored(tmp_path, caplog):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, QWEN_MMPROJ)  # Qwen projector next to a Gemma model

    with caplog.at_level(logging.WARNING, logger="Local Server Manager"):
        assert find_mmproj_sibling(models / GEMMA_MODEL) is None

    assert any("different model family" in rec.message for rec in caplog.records)
    assert any(QWEN_MMPROJ in rec.message for rec in caplog.records)


def test_no_projector_anywhere_returns_none(tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_EXTRA_QUANT)

    assert find_mmproj_sibling(models / GEMMA_MODEL) is None


def test_best_family_match_wins_when_several_projectors_present(tmp_path):
    models = tmp_path / "models"
    _touch(
        models,
        GEMMA_MODEL,
        GEMMA_PROJECTOR,
        "Huihui-gemma-4-2B-it-abliterated-projector.gguf",
    )

    assert find_mmproj_sibling(models / GEMMA_MODEL) == str(models / GEMMA_PROJECTOR)


# --- llm.yaml override ------------------------------------------------------


def test_llm_yaml_override_wins_over_sibling(manager, tmp_path, config_root):
    models = tmp_path / "models"
    explicit = models / "explicit-projector.gguf"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR, "explicit-projector.gguf")
    (config_root / "llm.yaml").write_text(f"mmproj_path: {explicit}\n")

    resolved = manager.resolve_mmproj_path(str(models / GEMMA_MODEL))

    assert resolved == str(explicit)


def test_llm_yaml_override_missing_file_falls_back_to_sibling(manager, tmp_path, config_root):
    (config_root / "llm.yaml").write_text(
        "mmproj_path: /nope/missing-projector.gguf\n"
    )
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR)

    assert manager.resolve_mmproj_path(str(models / GEMMA_MODEL)) == str(models / GEMMA_PROJECTOR)


def test_empty_mmproj_path_keeps_auto_detection(manager, tmp_path):
    # Default llm.yaml seeds mmproj_path: "" - sibling detection still runs.
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR)

    assert manager.resolve_mmproj_path(str(models / GEMMA_MODEL)) == str(models / GEMMA_PROJECTOR)


def test_no_mmproj_anywhere_returns_none(manager, tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL)

    assert manager.resolve_mmproj_path(str(models / GEMMA_MODEL)) is None


def test_override_supports_user_tilde(manager, tmp_path, config_root, monkeypatch):
    home = tmp_path / "home"
    _touch(home, "vision-projector.gguf")
    monkeypatch.setenv("HOME", str(home))
    (config_root / "llm.yaml").write_text("mmproj_path: ~/vision-projector.gguf\n")

    assert manager.resolve_mmproj_path(str(home / "model.gguf")) == str(home / "vision-projector.gguf")


# --- command building -------------------------------------------------------


class _FakeStream:
    async def readline(self):
        return b""


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()


def _make_bundled(module_root, profile="cpu"):
    binary = module_root / "backend" / profile / "llama-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    return binary


def _run_start_and_capture_command(manager, monkeypatch):
    captured: list[list[str]] = []

    async def fake_exec(*command, **kwargs):
        captured.append(list(command))
        return _FakeProcess()

    monkeypatch.setattr(lsm.asyncio, "create_subprocess_exec", fake_exec)
    manager.configuration_settings.update_main_setting("llm_device", 0)
    asyncio.run(manager.start_server_async())
    return captured[0]


def test_command_includes_mmproj_flag(manager, module_root, tmp_path, monkeypatch):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    command = _run_start_and_capture_command(manager, monkeypatch)

    assert "--mmproj" in command
    assert command[command.index("--mmproj") + 1] == str(models / GEMMA_PROJECTOR)


def test_command_without_mmproj_stays_text_only(manager, module_root, tmp_path, monkeypatch):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_EXTRA_QUANT)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    command = _run_start_and_capture_command(manager, monkeypatch)

    assert "--mmproj" not in command


def test_command_build_logs_multimodal_projector_line(
    manager, module_root, tmp_path, monkeypatch, caplog
):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_PROJECTOR)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    with caplog.at_level(logging.INFO, logger="Local Server Manager"):
        command = _run_start_and_capture_command(manager, monkeypatch)

    projector_line = f"Multimodal projector: {models / GEMMA_PROJECTOR}"
    assert any(rec.message == projector_line for rec in caplog.records)
    assert "--mmproj" in command
