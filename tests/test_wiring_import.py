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
