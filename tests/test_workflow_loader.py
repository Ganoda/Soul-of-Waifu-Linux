import json
from pathlib import Path

import pytest

from app.utils.media import workflow_loader


@pytest.fixture
def wf_root(tmp_path, monkeypatch):
    root = tmp_path / "workflows"
    (root / "tts").mkdir(parents=True)
    (root / "image").mkdir()
    (root / "video").mkdir()
    monkeypatch.setattr(workflow_loader, "_workflows_dir_override", str(root))
    yield root


def _write_api_workflow(path: Path, text_id="3", out_id="9"):
    graph = {
        text_id: {
            "class_type": "PrimitiveString",
            "inputs": {"value": ""},
            "_meta": {"title": "Text"},
        },
        out_id: {
            "class_type": "SaveAudio",
            "inputs": {},
            "_meta": {"title": "Save"},
        },
    }
    path.write_text(json.dumps(graph))


API_FORMAT = ("files must parse as json")

# --- fixtures under tests/fixtures/workflows used by tests below ---


def test_scan_finds_spec_with_meta_ids(tmp_path, wf_root):
    p = wf_root / "tts" / "my_voice.json"
    _write_api_workflow(p)
    meta = {
        "text_node_id": "3",
        "output_node_id": "9",
    }
    (wf_root / "tts" / "my_voice.meta.yaml").write_text(
        "text_node_id: '3'\noutput_node_id: '9'\n"
    )

    specs = workflow_loader.scan("tts")

    assert len(specs) == 1
    s = specs[0]
    assert s.id == "my_voice"
    assert s.category == "tts"
    assert s.text_node_id == "3"
    assert s.output_node_id == "9"
    assert s.source_meta is True
    assert isinstance(s.raw, dict)


def test_heuristic_detection_without_meta(wf_root):
    _write_api_workflow(wf_root / "tts" / "auto.json")
    # keep output SaveAudio (regex-detectable); text node PrimitiveString is heuristic-detected
    specs = workflow_loader.scan("tts")

    assert len(specs) == 1
    s = specs[0]
    assert s.source_meta is False
    assert s.text_node_id == "3"
    assert s.output_node_id == "9"


def test_malformed_json_skipped_sibling_still_found(wf_root):
    (wf_root / "image" / "broken.json").write_text("{ not json ]")
    _write_api_workflow(wf_root / "image" / "good.json")

    specs = workflow_loader.scan("image")

    assert [s.id for s in specs] == ["good"]


def test_meta_extra_unknown_key_ignored(wf_root):
    _write_api_workflow(wf_root / "tts" / "extra.json")
    (wf_root / "tts" / "extra.meta.yaml").write_text(
        "text_node_id: '3'\noutput_node_id: '9'\nbogus_key: whatever\n"
    )

    specs = workflow_loader.scan("tts")

    assert len(specs) == 1
    assert specs[0].text_node_id == "3"


def test_meta_bad_key_defaulted_to_heuristic(wf_root):
    _write_api_workflow(wf_root / "tts" / "badmeta.json", text_id="5", out_id="7")
    (wf_root / "tts" / "badmeta.meta.yaml").write_text(
        "text_node_id: ''\noutput_node_id: ''\n"
    )
    # make heuristic-detectable: rename classes
    p = wf_root / "tts" / "badmeta.json"
    g = json.loads(p.read_text())
    g["5"]["class_type"] = "CLIPTextEncode"
    g["5"]["inputs"] = {"text": ""}
    g["7"]["class_type"] = "SaveImage"
    p.write_text(json.dumps(g))

    specs = workflow_loader.scan("tts")

    s = specs[0]
    assert s.text_node_id == "5"
    assert s.output_node_id == "7"
    assert s.source_meta is False


def test_empty_dir_returns_empty_list(wf_root):
    assert workflow_loader.scan("video") == []


def test_input_field_detected_for_heuristic_text_node(wf_root):
    p = wf_root / "tts" / "fieldcheck.json"
    graph = {
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "4": {"class_type": "SaveImage", "inputs": {}},
    }
    p.write_text(json.dumps(graph))

    specs = workflow_loader.scan("tts")

    assert specs[0].text_input_field == "text"
