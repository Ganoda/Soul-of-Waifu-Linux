"""XDG-based configuration root resolution for Soul of Waifu (Linux port).

Resolution order:
  1. $SOW_CONFIG_DIR
  2. $XDG_CONFIG_HOME/soul-of-waifu
  3. ~/.config/soul-of-waifu

Legacy JSON files living in app/configuration/ are copied to the resolved
root on first use; existing files at the target are never overwritten.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_legacy_names = ("settings.json", "api.json", "characters.json")
_repo_dir_hint = ""
_migrated = False


def _repo_root() -> Path:
    if _repo_dir_hint:
        return Path(_repo_dir_hint)
    return Path(__file__).resolve().parents[2]


def config_root() -> Path:
    env_dir = os.environ.get("SOW_CONFIG_DIR")
    if env_dir:
        return Path(env_dir).expanduser()
    # Windows (upstream builds) keeps legacy per-repo JSON location;
    # Linux/Wayland uses the XDG hierarchy.
    if sys.platform == "win32":
        return repo_config_dir()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "soul-of-waifu"


def repo_config_dir() -> Path:
    return _repo_root() / "app" / "configuration"


def config_file(name: str) -> Path:
    return config_root() / name


def ensure_xdg_layout(migrate: bool = True) -> bool:
    """Create the XDG config root and seed legacy files.

    Returns True when any file was copied, False otherwise. Idempotent:
    guarded by a module-level flag after the first invocation, except when
    seeding actually happened or ``migrate=False`` forces a re-check.
    """
    global _migrated

    root = config_root()
    root.mkdir(parents=True, exist_ok=True)

    if not migrate or _migrated or sys.platform == "win32":
        return False

    repo_cfg = repo_config_dir()
    if not repo_cfg.is_dir():
        _migrated = True
        return False

    copied_any = False
    for name in _legacy_names:
        src = repo_cfg / name
        dst = root / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            logger.info("[Config] migrated %s -> %s", src, dst)
            copied_any = True

    _migrated = True
    return copied_any
