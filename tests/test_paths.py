import os
import shutil

import pytest

from app.configuration import paths


@pytest.fixture(autouse=True)
def fresh_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("SOW_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(paths, "_migrated", False)
    yield tmp_path


def test_default_root_is_xdg(fresh_paths):
    root = paths.config_root()
    assert root == pathlib_home() / ".config" / "soul-of-waifu"


def pathlib_home():
    import pathlib
    return pathlib.Path.home()


def test_sow_config_dir_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("SOW_CONFIG_DIR", str(tmp_path / "custom"))
    assert paths.config_root() == tmp_path / "custom"


def test_xdg_config_home_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert paths.config_root() == tmp_path / "xdg" / "soul-of-waifu"


def test_config_file_joins_name(fresh_paths):
    p = paths.config_file("settings.json")
    assert p.name == "settings.json"
    assert p.parent == paths.config_root()


LEGACY_FILES = ("settings.json", "api.json", "characters.json")


def _make_legacy_repo(repo: str) -> None:
    os.makedirs(os.path.join(repo, "app", "configuration"), exist_ok=True)
    for name in LEGACY_FILES:
        with open(os.path.join(repo, "app", "configuration", name), "w") as f:
            f.write('{"legacy": true}')


def test_ensure_xdg_layout_copies_missing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_legacy_repo(str(repo))
    monkeypatch.setattr(paths, "_repo_dir_hint", str(repo))
    target = tmp_path / "cfg"
    monkeypatch.setenv("SOW_CONFIG_DIR", str(target))

    copied = paths.ensure_xdg_layout()

    assert copied is True
    for name in LEGACY_FILES:
        assert (target / name).read_text() == '{"legacy": true}'


def test_ensure_xdg_layout_preserves_existing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    _make_legacy_repo(str(repo))
    monkeypatch.setattr(paths, "_repo_dir_hint", str(repo))
    target = tmp_path / "cfg"
    target.mkdir()
    (target / "settings.json").write_text('{"user": true}')
    monkeypatch.setenv("SOW_CONFIG_DIR", str(target))
    monkeypatch.setattr(paths, "_migrated", False)

    copied = paths.ensure_xdg_layout()

    assert (target / "settings.json").read_text() == '{"user": true}'
    # api.json and characters.json were missing, they are still seeded
    assert (target / "api.json").exists()
    assert (target / "characters.json").exists()


def test_ensure_xdg_layout_idempotent_when_nothing_to_do(tmp_path, monkeypatch):
    target = tmp_path / "cfg"
    monkeypatch.setenv("SOW_CONFIG_DIR", str(target))
    empty_repo = tmp_path / "emptyrepo"
    (empty_repo / "app" / "configuration").mkdir(parents=True)
    monkeypatch.setattr(paths, "_repo_dir_hint", str(empty_repo))

    assert paths.ensure_xdg_layout() is False
    assert paths.ensure_xdg_layout() is False  # flag guard: no repeated work
