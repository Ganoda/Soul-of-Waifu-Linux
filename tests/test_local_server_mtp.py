"""MTP (multi-token prediction) draft support for the embedded llama-server.

Covers sibling auto-detection next to the selected GGUF (including the
merge-aware match for differently hyphenated names like "Gemma4-12B-QAT..."
vs "mtp-gemma-4-12B-it"), the llm.yaml mtp_path override with the explicit
"off" switch, the binary support probe, and the "--spec-type/--model-draft"
flag wiring in the built command.
"""

import asyncio
import logging

import pytest

from app.utils import xconfig
from app.utils.ai_clients import local_server_manager as lsm
from app.utils.ai_clients.local_server_manager import (
    find_mtp_sibling,
    probe_mtp_spec_type,
)
from app.utils.ai_clients.local_server_manager import LocalServerManager


GEMMA_MODEL = "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf"
GEMMA_MTP = "mtp-gemma-4-12B-it.gguf"  # hyphenated differently from the model
GEMMA_MMPROJ = "mmproj-Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf"
GEMMA_2B_MTP = "mtp-gemma-4-2B-it.gguf"  # same family line, wrong size
FOREIGN_MTP = "mtp-q2_0.gguf"
QWEN_MODEL = "Swift-Qwen3.8-Flash-Next-Q4_K_M.gguf"
QWEN_MTP_SUFFIX = "Swift-Qwen3.8-Flash-Next-MTP.gguf"


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg))
    xconfig.reset_cache()
    lsm._MTP_SPEC_SUPPORT_CACHE.clear()
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


def test_detects_merge_matched_mtp_next_to_model(tmp_path):
    # The real-world case: the draft spells "gemma-4-12B" while the model
    # spells "Gemma4-12B"; plain token-prefix/shared scoring finds nothing.
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP, GEMMA_MMPROJ)

    assert find_mtp_sibling(models / GEMMA_MODEL) == str(models / GEMMA_MTP)


def test_mtp_scanner_ignores_mmproj_files_and_vice_versa(tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP, GEMMA_MMPROJ)

    # find_mtp_sibling must not grab the projector...
    assert find_mtp_sibling(models / GEMMA_MODEL) == str(models / GEMMA_MTP)
    # ...and the projector scanner (unchanged behavior) still finds the mmproj.
    from app.utils.ai_clients.local_server_manager import find_mmproj_sibling

    assert find_mmproj_sibling(models / GEMMA_MODEL) == str(models / GEMMA_MMPROJ)


def test_wrong_size_draft_of_same_family_is_ignored(tmp_path, caplog):
    # "gemma-4-2B" merges only partially against "Gemma4-12B" ("2b" vs "12b"):
    # attaching it would abort the launch, so it must be skipped.
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_2B_MTP)

    with caplog.at_level(logging.WARNING, logger="Local Server Manager"):
        assert find_mtp_sibling(models / GEMMA_MODEL) is None

    assert any("different model family" in rec.message for rec in caplog.records)
    assert any(GEMMA_2B_MTP in rec.message for rec in caplog.records)


def test_foreign_family_mtp_is_ignored(tmp_path, caplog):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, FOREIGN_MTP)

    with caplog.at_level(logging.WARNING, logger="Local Server Manager"):
        assert find_mtp_sibling(models / GEMMA_MODEL) is None

    assert any(FOREIGN_MTP in rec.message for rec in caplog.records)


def test_detects_suffix_style_mtp(tmp_path):
    models = tmp_path / "models"
    _touch(models, QWEN_MODEL, QWEN_MTP_SUFFIX)

    assert find_mtp_sibling(models / QWEN_MODEL) == str(models / QWEN_MTP_SUFFIX)


def test_no_mtp_anywhere_returns_none(tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MMPROJ)

    assert find_mtp_sibling(models / GEMMA_MODEL) is None


def test_best_family_match_wins_when_several_mtp_present(tmp_path):
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP, FOREIGN_MTP)

    assert find_mtp_sibling(models / GEMMA_MODEL) == str(models / GEMMA_MTP)


# --- llm.yaml override ------------------------------------------------------


def test_llm_yaml_override_wins_over_sibling(manager, tmp_path, config_root):
    models = tmp_path / "models"
    explicit = models / "explicit-mtp.gguf"
    _touch(models, GEMMA_MODEL, GEMMA_MTP, "explicit-mtp.gguf")
    (config_root / "llm.yaml").write_text(f"mtp_path: {explicit}\n")

    resolved = manager.resolve_mtp_path(str(models / GEMMA_MODEL))

    assert resolved == str(explicit)


def test_llm_yaml_off_disables_mtp_even_when_sibling_exists(manager, tmp_path, config_root, caplog):
    # A bare `off` is parsed as boolean False by YAML 1.1 - and quoted "off"
    # stays a string; both must disable MTP.
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)

    (config_root / "llm.yaml").write_text("mtp_path: off\n")
    assert manager.resolve_mtp_path(str(models / GEMMA_MODEL)) is None

    xconfig.reset_cache()
    (config_root / "llm.yaml").write_text('mtp_path: "off"\n')
    assert manager.resolve_mtp_path(str(models / GEMMA_MODEL)) is None


def test_llm_yaml_override_missing_file_falls_back_to_sibling(manager, tmp_path, config_root):
    (config_root / "llm.yaml").write_text("mtp_path: /nope/missing-mtp.gguf\n")
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)

    assert manager.resolve_mtp_path(str(models / GEMMA_MODEL)) == str(models / GEMMA_MTP)


def test_empty_mtp_path_keeps_auto_detection(manager, tmp_path):
    # Default llm.yaml seeds mtp_path: "" - sibling detection still runs.
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)

    assert manager.resolve_mtp_path(str(models / GEMMA_MODEL)) == str(models / GEMMA_MTP)


def test_override_supports_user_tilde(manager, tmp_path, config_root, monkeypatch):
    home = tmp_path / "home"
    _touch(home, "accelerator-mtp.gguf")
    monkeypatch.setenv("HOME", str(home))
    (config_root / "llm.yaml").write_text("mtp_path: ~/accelerator-mtp.gguf\n")

    assert manager.resolve_mtp_path(str(home / "model.gguf")) == str(home / "accelerator-mtp.gguf")


# --- binary support probe ---------------------------------------------------


def _write_script(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def test_probe_returns_draft_mtp_for_current_builds(tmp_path):
    binary = _write_script(
        tmp_path / "server-a",
        "echo '--spec-type none,draft-simple,draft-mtp,draft-eagle3'",
    )

    assert probe_mtp_spec_type(str(binary)) == "draft-mtp"


def test_probe_returns_mtp_for_older_spelling(tmp_path):
    binary = _write_script(
        tmp_path / "server-b",
        "echo '--spec-type none,simple,mtp,eagle'",
    )

    assert probe_mtp_spec_type(str(binary)) == "mtp"


def test_probe_returns_none_for_build_without_mtp(tmp_path):
    binary = _write_script(
        tmp_path / "server-c",
        "echo '--spec-type none,simple,eagle'",
    )

    assert probe_mtp_spec_type(str(binary)) is None


def test_probe_is_cached_per_binary(tmp_path):
    binary = _write_script(tmp_path / "server-d", "echo 'draft-mtp'")
    first = probe_mtp_spec_type(str(binary))

    binary.write_text("#!/bin/sh\nexit 1\n")  # a changed binary is not re-probed

    assert probe_mtp_spec_type(str(binary)) == first


def test_probe_survives_unexecutable_file(tmp_path):
    binary = tmp_path / "server-e"
    binary.write_text("not a script")
    binary.chmod(0o644)

    assert probe_mtp_spec_type(str(binary)) is None


# --- command building -------------------------------------------------------


class _FakeStream:
    async def readline(self):
        return b""


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()


def _make_bundled(module_root, help_text="--spec-type none,draft-simple,draft-mtp"):
    binary = module_root / "backend" / "cpu" / "llama-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(f"#!/bin/sh\necho '{help_text}'\n")
    binary.chmod(0o755)
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


def test_command_includes_mtp_flags(manager, module_root, tmp_path, monkeypatch):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    command = _run_start_and_capture_command(manager, monkeypatch)

    assert "--spec-type" in command
    assert command[command.index("--spec-type") + 1] == "draft-mtp"
    assert "--model-draft" in command
    assert command[command.index("--model-draft") + 1] == str(models / GEMMA_MTP)


def test_command_keeps_mmproj_and_mtp_together(manager, module_root, tmp_path, monkeypatch):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP, GEMMA_MMPROJ)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    command = _run_start_and_capture_command(manager, monkeypatch)

    assert command[command.index("--mmproj") + 1] == str(models / GEMMA_MMPROJ)
    assert command[command.index("--model-draft") + 1] == str(models / GEMMA_MTP)


def test_command_without_mtp_support_stays_plain(manager, module_root, tmp_path, monkeypatch, caplog):
    # A build without MTP must still launch - just without the draft.
    _make_bundled(module_root, help_text="--spec-type none,simple")
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    with caplog.at_level(logging.WARNING, logger="Local Server Manager"):
        command = _run_start_and_capture_command(manager, monkeypatch)

    assert "--spec-type" not in command
    assert "--model-draft" not in command
    assert any("does not support MTP" in rec.message for rec in caplog.records)


def test_command_without_mtp_sibling_stays_plain(manager, module_root, tmp_path, monkeypatch):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    command = _run_start_and_capture_command(manager, monkeypatch)

    assert "--spec-type" not in command
    assert "--model-draft" not in command


def test_command_build_logs_mtp_draft_line(manager, module_root, tmp_path, monkeypatch, caplog):
    _make_bundled(module_root)
    models = tmp_path / "models"
    _touch(models, GEMMA_MODEL, GEMMA_MTP)
    manager.configuration_settings.update_main_setting("local_llm", str(models / GEMMA_MODEL))

    with caplog.at_level(logging.INFO, logger="Local Server Manager"):
        command = _run_start_and_capture_command(manager, monkeypatch)

    draft_line = f"MTP draft model: {models / GEMMA_MTP} (--spec-type draft-mtp)"
    assert any(rec.message == draft_line for rec in caplog.records)
    assert "--model-draft" in command
