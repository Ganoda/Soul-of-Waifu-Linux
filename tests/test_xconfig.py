import os

import pytest

from app.utils import xconfig


@pytest.fixture(autouse=True)
def config_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SOW_CONFIG_DIR", str(tmp_path / "cfg"))
    xconfig.reset_cache()
    yield tmp_path / "cfg"


def test_load_seeds_defaults_from_repo(config_root):
    cfg = xconfig.load("media")
    assert (config_root / "media.yaml").exists()
    assert cfg.get("comfyui.base_url") == "http://127.0.0.1:8188"


def test_reload_picks_up_external_edit(config_root):
    xconfig.load("media")
    p = config_root / "media.yaml"
    text = p.read_text().replace(
        "base_url: http://127.0.0.1:8188",
        "base_url: http://127.0.0.1:9999",
    )
    p.write_text(text)

    cfg = xconfig.reload("media")
    assert cfg.get("comfyui.base_url") == "http://127.0.0.1:9999"


def test_get_unknown_dotted_returns_default(config_root):
    cfg = xconfig.load("ui")
    assert cfg.get("does.not.exist", 42) == 42
    assert cfg.get("does.not.exist") is None


def test_corrupt_yaml_falls_back_to_defaults(config_root):
    (config_root).mkdir(parents=True, exist_ok=True)
    (config_root / "llm.yaml").write_text("{ this is: not: valid")
    cfg = xconfig.load("llm")
    # Empty backend by default: means "use the embedded llama-server launcher"
    assert cfg.get("backend") == ""


def test_user_edited_value_survives_new_session(config_root):
    xconfig.load("ui")
    p = config_root / "ui.yaml"
    text = p.read_text().replace("theme: system", "theme: dark")
    p.write_text(text)
    xconfig.reset_cache()

    cfg = xconfig.load("ui")
    assert cfg.get("theme") == "dark"
