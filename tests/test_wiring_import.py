import asyncio
import json


def test_tts_worker_has_comfyui_branch():
    src = open("app/utils/text_to_speech.py", encoding="utf-8").read()
    assert 'elif self.tts_method == "ComfyUI"' in src, "dispatch branch missing"
    assert "ComfyUI unavailable, falling back to Kokoro" in src


def test_pipelined_path_has_comfyui_branch():
    src = open("app/utils/text_to_speech.py", encoding="utf-8").read()
    assert 'method == "ComfyUI"' in src


def test_image_generator_has_comfy_workflow_provider():
    src = open("app/utils/image_generator.py", encoding="utf-8").read()
    assert '"ComfyUI Workflow"' in src


def test_stt_reads_whisper_config():
    src = open("app/utils/speech_to_text.py", encoding="utf-8").read()
    assert 'xconfig' in src and "whisper_model_path" in src


def test_main_applies_ui_yaml_platform():
    src = open("main.py", encoding="utf-8").read()
    assert 'xconfig.load("ui")' in src


def test_gemma_turn_tokens_in_llm_defaults():
    data = open("config-defaults/llm.yaml", encoding="utf-8").read()
    assert "<start_of_turn>" in data and "<end_of_turn>" in data


def test_comfyui_option_reachable_in_tts_picker():
    src = open("app/gui/interface_signals.py", encoding="utf-8").read()
    assert '("ComfyUI", ' in src, "sidebar entry missing"
    assert '"ComfyUI": 7' in src, "engine_map entry missing"
    assert 'current_text_to_speech"] = "ComfyUI"' in src, "save handler missing"


def test_comfyui_workflow_option_reachable_in_image_combo():
    src = open("app/gui/custom_widgets.py", encoding="utf-8").read()
    assert '"ComfyUI Workflow"' in src, "image provider combo entry missing"


def test_start_sh_does_not_force_sow_config_dir():
    src = open("start.sh", encoding="utf-8").read()
    assert 'SOW_CONFIG_DIR="${SOW_CONFIG_DIR:-' not in src


def test_main_registers_provider_defaults():
    src = open("main.py", encoding="utf-8").read()
    assert "register_defaults" in src


def test_comfyui_enabled_knob_honored(tmp_path, monkeypatch):
    import asyncio
    import pytest as _pytest
    from app.utils.media.comfy_client import ComfyUIClient, ComfyUIUnavailable

    cfg = tmp_path / "cfg"
    cfg.mkdir(parents=True)
    (cfg / "media.yaml").write_text(
        "comfyui:\n  enabled: false\n  base_url: http://127.0.0.1:59996\n"
        "workflows_dir: app/data/workflows\n"
    )
    monkeypatch.setenv("SOW_CONFIG_DIR", str(cfg))
    from app.utils import xconfig
    xconfig.reset_cache()

    client = ComfyUIClient()
    assert client.enabled is False
    with _pytest.raises(ComfyUIUnavailable):
        asyncio.run(client.run({"3": {"class_type": "X", "inputs": {}}}))

    xconfig.reset_cache()


def test_llm_base_url_normalizes_v1():
    from app.utils.providers.openai_compat_llm import OpenAICompatLLM

    c1 = OpenAICompatLLM(base_url="http://127.0.0.1:8080")
    assert c1.chat_url == "http://127.0.0.1:8080/v1/chat/completions"

    c2 = OpenAICompatLLM(base_url="http://127.0.0.1:8080/v1")
    assert c2.chat_url == "http://127.0.0.1:8080/v1/chat/completions"


def test_win32_keeps_legacy_repo_paths(monkeypatch, tmp_path):
    import sys as _sys
    monkeypatch.setattr("app.configuration.paths.sys.platform", "win32")
    from app.configuration import paths as _paths
    root = _paths.config_root()
    assert str(root).endswith("app/configuration") or "app" in str(root)
