"""Embedded llama-server discovery: bundled binary -> system PATH -> clear error.

Covers the "empty llm.yaml backend and no win-binaries shipped" scenario on
Linux: the embedded launcher must fall back to a distro-installed llama-server
found on PATH and produce an actionable, translated error when it cannot.
"""

import asyncio

import pytest

from app.utils import xconfig
from app.utils.ai_clients import local_server_manager as lsm
from app.utils.ai_clients.local_server_manager import LocalServerManager
from app.configuration import configuration


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg))
    xconfig.reset_cache()
    yield cfg


@pytest.fixture(autouse=True)
def clean_error_state():
    lsm.clear_embedded_server_error()
    yield
    lsm.clear_embedded_server_error()


@pytest.fixture
def module_root(tmp_path, monkeypatch):
    """Fake app/utils/ai_clients dir anchoring the backend/ subtree."""
    monkeypatch.setattr(lsm, "MODULE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def manager(module_root):
    return LocalServerManager(ui=None)


def _make_bundled(module_root, profile="vulkan"):
    binary = module_root / "backend" / profile / "llama-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"#!/bin/sh\nexit 0\n")
    return binary


def _select_profile(device, backend):
    cs = configuration.ConfigurationSettings()
    cs.update_main_setting("llm_device", device)
    cs.update_main_setting("llm_backend", backend)


def test_bundled_vulkan_binary_preferred(module_root, manager):
    binary = _make_bundled(module_root, "vulkan")

    assert manager.resolve_server_executable(1, 0) == str(binary)


def test_bundled_cuda_binary_per_backend_selection(module_root, manager):
    binary = _make_bundled(module_root, "cuda")
    _select_profile(1, 1)

    assert manager.resolve_server_executable(1, 1) == str(binary)


def test_cpu_device_uses_cpu_profile(module_root, manager):
    binary = _make_bundled(module_root, "cpu")
    _select_profile(0, 0)

    assert manager.resolve_server_executable(0, 0) == str(binary)


def test_windows_exe_suffix_applied_only_on_win32(module_root, manager, monkeypatch):
    monkeypatch.setattr(lsm.platform, "system", lambda: "Windows")
    binary = _make_bundled(module_root, "vulkan")
    exe = binary.with_name("llama-server.exe")
    binary.rename(exe)

    resolved = manager.resolve_server_executable(1, 0)
    assert resolved == str(exe)

    monkeypatch.setattr(lsm.platform, "system", lambda: "Linux")
    binary.write_bytes(exe.read_bytes())
    assert manager.resolve_server_executable(1, 0) == str(binary)


def test_missing_bundled_falls_back_to_system_path_binary(module_root, manager, tmp_path, monkeypatch):
    fake_system = tmp_path / "usr" / "bin" / "llama-server"
    fake_system.parent.mkdir(parents=True, exist_ok=True)
    fake_system.write_bytes(b"#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(lsm.shutil, "which", lambda name: str(fake_system))

    resolved = manager.locate_server_binary(manager.resolve_server_executable(1, 0))

    assert resolved == str(fake_system)


def test_gpu_flags_plan_applies_to_system_binary_the_same_way(module_root, manager, tmp_path, monkeypatch):
    # Profile selection stays orthogonal to where the binary came from:
    # the same runner drives either binary with the settings-driven flags.
    fake_system = tmp_path / "bin" / "llama-server"
    monkeypatch.setattr(lsm.shutil, "which", lambda name: str(fake_system))
    _select_profile(1, 1)  # cuda profile despite system binary

    resolved = manager.locate_server_binary(manager.resolve_server_executable(1, 1))
    assert resolved == str(fake_system)


def test_not_found_anywhere_raises_actionable_error(module_root, manager, monkeypatch):
    monkeypatch.setattr(lsm.shutil, "which", lambda name: None)
    _select_profile(1, 0)

    with pytest.raises(RuntimeError) as err:
        asyncio.run(manager.start_server_async())

    message = str(err.value)
    assert "llama-server" in message
    assert "llama.cpp" in message
    assert "backend" in message or "PATH" in message
    # Root cause is published for the provider/UI hint layer
    assert lsm.recent_embedded_server_error() == message


def test_ensure_server_running_propagates_discovery_error(
    module_root, manager, monkeypatch
):
    monkeypatch.setattr(lsm.shutil, "which", lambda name: None)
    monkeypatch.setattr(manager, "is_server_already_running", lambda: False)
    _select_profile(1, 0)

    with pytest.raises(RuntimeError) as err:
        asyncio.run(manager.ensure_server_running())

    assert "llama-server" in str(err.value)


def test_recording_is_non_consumer_with_freshness_ttl(monkeypatch):
    lsm.record_embedded_server_error("boom")

    assert lsm.recent_embedded_server_error() == "boom"
    assert lsm.recent_embedded_server_error() == "boom"  # peek, not consume

    monkeypatch.setattr(lsm, "_error_recorded_at", lsm._error_recorded_at - 301)
    assert lsm.recent_embedded_server_error() is None

    lsm.clear_embedded_server_error()
    assert lsm.recent_embedded_server_error() is None


def test_provider_connection_error_carries_root_cause(module_root, monkeypatch):
    from app.utils.ai_clients.providers.local_provider import LocalProvider

    monkeypatch.setattr(lsm.shutil, "which", lambda name: None)
    _select_profile(1, 0)
    manager = LocalServerManager(ui=None)
    monkeypatch.setattr(manager, "is_server_already_running", lambda: False)

    with pytest.raises(RuntimeError):
        asyncio.run(manager.ensure_server_running())

    provider = LocalProvider()
    hint = provider._connection_error_message(ConnectionError("refused"))

    assert "Local API Error" in hint
    assert "llama-server" in hint
    assert "pacman -S llama.cpp" in hint


def test_provider_connection_error_without_recorded_cause_stays_clean(module_root):
    from app.utils.ai_clients.providers.local_provider import LocalProvider

    provider = LocalProvider()
    hint = provider._connection_error_message(ConnectionError("refused"))

    assert "Local API Error" in hint
    assert "↳" not in hint
