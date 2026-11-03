"""Modular YAML configuration layer.

Each module (llm, media, ui) is a single YAML file at the config root
(resolved by app.configuration.paths). On first access the file is seeded
from the in-repo defaults (config-defaults/<module>.yaml); user edits then
win until the file is deleted. Corrupt user files fall back to defaults
with a logged warning — never an exception.
"""

import logging
from pathlib import Path

import yaml

from app.configuration import paths

logger = logging.getLogger("XConfig")

_repo_root = Path(__file__).resolve().parents[2]
_cache: dict[str, "XConfig"] = {}


class XConfig:
    def __init__(self, module: str, data: dict):
        self._module = module
        self._data = data or {}

    def get(self, dotted: str, default=None):
        node = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def as_dict(self) -> dict:
        return self._data


def _defaults_dir() -> Path:
    return _repo_root / "config-defaults"


def reset_cache() -> None:
    _cache.clear()


def load(module: str) -> XConfig:
    if module in _cache:
        return _cache[module]

    root = paths.config_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{module}.yaml"

    if not target.exists():
        src = _defaults_dir() / f"{module}.yaml"
        try:
            target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        except FileNotFoundError:
            logger.warning("[XConfig] no defaults for module %r", module)

    cfg = reload(module)
    _cache[module] = cfg
    return cfg


def reload(module: str) -> XConfig:
    root = paths.config_root()
    target = root / f"{module}.yaml"

    data: dict = {}
    if target.exists():
        try:
            data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError("not a mapping")
        except (yaml.YAMLError, ValueError) as e:
            logger.warning("[XConfig] %s is corrupt (%s), using built-in defaults", target, e)
            data = _builtin_defaults(module)
    else:
        data = _builtin_defaults(module)

    cfg = XConfig(module, data)
    _cache[module] = cfg
    return cfg


def _builtin_defaults(module: str) -> dict:
    try:
        return yaml.safe_load(
            (_defaults_dir() / f"{module}.yaml").read_text(encoding="utf-8")
        ) or {}
    except FileNotFoundError:
        return {}
