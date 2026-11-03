"""Drop-in workflow discovery for ComfyUI API-format exports.

Category layout (from media.yaml workflows_dir):
    <workflows_dir>/<category>/*.json       — workflow graph (API format)
    <workflows_dir>/<category>/<name>.meta.yaml — optional node mapping

Meta keys: text_node_id, output_node_id. Meta wins over heuristics; blank
or invalid meta values fall back to heuristic detection.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.utils import xconfig

logger = logging.getLogger("WorkflowLoader")

_repo_root = Path(__file__).resolve().parents[2]

_workflows_dir_override = ""

_TEXT_INPUT_KEYS = ("text", "string", "value", "prompt")
_TEXT_CLASS_HINTS = ("primitivestring", "cliptextencode", "textto", "tts")
_OUTPUT_CLASS_RE = re.compile(r"Save(Audio|Image)|VHS_VideoCombine$", re.IGNORECASE)


@dataclass(frozen=True)
class WorkflowSpec:
    id: str
    path: Path
    category: str
    text_node_id: str | None
    text_input_field: str
    output_node_id: str | None
    source_meta: bool
    raw: dict


def _workflows_base() -> Path:
    if _workflows_dir_override:
        return Path(_workflows_dir_override)
    configured = xconfig.load("media").get("workflows_dir", "app/data/workflows")
    return (_repo_root / configured).resolve()


def scan(category: str, *, base_dir=None) -> list[WorkflowSpec]:
    base = Path(base_dir) if base_dir else _workflows_base()
    directory = base / category
    if not directory.is_dir():
        return []

    specs: list[WorkflowSpec] = []
    for json_path in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.info("[Workflows] skipping %s: %s", json_path.name, e)
            continue
        if not isinstance(raw, dict):
            logger.info("[Workflows] skipping %s: not a node map", json_path.name)
            continue

        meta_text, meta_out = _read_meta(json_path)
        text_id, input_field, source = _detect_text_node(raw, meta_text)
        out_id = meta_out or _detect_output_node(raw)

        specs.append(
            WorkflowSpec(
                id=json_path.stem,
                path=json_path,
                category=category,
                text_node_id=text_id,
                text_input_field=input_field or "text",
                output_node_id=out_id,
                source_meta=source,
                raw=raw,
            )
        )
    return specs


def _read_meta(json_path: Path) -> tuple[str | None, str | None]:
    meta_path = json_path.with_suffix(".meta.yaml")
    if not meta_path.exists():
        return None, None
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        logger.info("[Workflows] bad meta %s: %s", meta_path.name, e)
        return None, None
    if not isinstance(data, dict):
        return None, None
    text = data.get("text_node_id")
    out = data.get("output_node_id")
    return (
        str(text) if text not in (None, "") else None,
        str(out) if out not in (None, "") else None,
    )


def _node_title(node: dict) -> str:
    meta = node.get("_meta") or {}
    title = meta.get("title") if isinstance(meta, dict) else ""
    return title if isinstance(title, str) else ""


def _detect_text_node(raw: dict, meta_hint: str | None):
    if meta_hint and meta_hint in raw:
        node = raw[meta_hint]
        inputs = node.get("inputs") or {}
        field_key = next((k for k in _TEXT_INPUT_KEYS if k in inputs), "")
        return meta_hint, field_key, True

    for node_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type", ""))
        cls_l = cls.lower()
        matched = any(h in cls_l for h in ("primitivestring",))
        matched = matched or any(h in cls_l for h in ("cliptextencode",))
        matched = matched or ("textto" in cls_l) or ("tts" in cls_l)
        if not matched:
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        field_key = next((k for k in _TEXT_INPUT_KEYS if k in inputs), "")
        return node_id, field_key, False

    return None, "", False


def _detect_output_node(raw: dict):
    for node_id, node in raw.items():
        if not isinstance(node, dict):
            continue
        cls = str(node.get("class_type", ""))
        if _OUTPUT_CLASS_RE.search(cls):
            return node_id
    for node_id, node in raw.items():
        if isinstance(node, dict) and _OUTPUT_CLASS_RE.search(_node_title(node)):
            return node_id
    return None
