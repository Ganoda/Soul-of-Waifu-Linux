import os
import io
import re
import uuid
import queue
import time
import shutil
import concurrent.futures

# Must be set before torch initializes its CUDA allocator: the TTS models
# share the GPU with llama-server, whose compute buffers grow/shrink around
# them — without expandable segments the allocator fragments and OOMs early.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import asyncio
import logging
import edge_tts
import threading
import contextlib
import numpy as np
import soundfile as sf
import sounddevice as sd
from typing import Optional

from TTS.api import TTS
from kokoro import KPipeline
from pydub import AudioSegment
from elevenlabs.client import AsyncElevenLabs
from qwen_tts import Qwen3TTSModel

from PyQt6.QtCore import QThread, pyqtSignal

from app.configuration import configuration

try:
    from rvc_python.infer import RVCInference
except ImportError:
    RVCInference = None

import torch.serialization
try:
    from fairseq.data.dictionary import Dictionary
    torch.serialization.add_safe_globals([Dictionary])
except Exception:
    pass

logger = logging.getLogger("Text-To-Speech Module")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, "app", "models", "hf_cache")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

os.makedirs(CACHE_DIR, exist_ok=True)


def log_tts_engine_fallback(owner, method, next_method, exc=None):
    """Log a fallback hop of the tts_engine_order chain (deduplicated).

    Keeps the historical ComfyUI->Kokoro wording so existing log users and
    wiring expectations stay recognizable.
    """
    warned = getattr(owner, "_fallback_warned", None)
    if warned is None:
        warned = set()
        owner._fallback_warned = warned
    key = (method, next_method, exc is None)
    level = logging.WARNING if key not in warned else logging.DEBUG
    warned.add(key)
    if method == "ComfyUI" and next_method == "Kokoro":
        detail = f": {exc}" if exc else ""
        logger.log(level, "[Media] ComfyUI workflow produced no audio, falling back to Kokoro%s"
                   if exc is None else "[Media] ComfyUI unavailable, falling back to Kokoro%s", detail)
    else:
        detail = f" ({exc})" if exc else ""
        logger.log(level, "[Media] TTS engine %r gave no audio%s, trying next engine %r from tts_engine_order",
                   method, detail, next_method)


def fix_rvc_sample_rate(file_path, base_file_path=None, target_sr=48000):
    if not file_path or not os.path.exists(file_path):
        return file_path
    try:
        data, sr = sf.read(file_path, dtype='float32')

        if sr != target_sr:
            if base_file_path and os.path.exists(base_file_path):
                data_base, sr_base = sf.read(base_file_path, dtype='float32')
                dur_base = len(data_base) / sr_base
                if dur_base > 0:
                    target_sr = int(round(len(data) / dur_base))

            sf.write(file_path, data, target_sr)
            logger.info(f"[RVC Fix] Corrected header from {sr} Hz to {target_sr} Hz")
    except Exception as e:
        logger.warning(f"Failed to fix RVC sample rate: {e}")
    return file_path

class ElevenLabs:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.audio_cache = AudioSegment.empty()
        self.device_index = self.configuration_settings.get_main_setting("output_device_real_index")
        self.output_dir = "app/voices/elevenlabs_audio"
        os.makedirs(self.output_dir, exist_ok=True)

    async def generate_speech_with_elevenlabs(self, text, voice_id):
        try:
            self.eleven_labs_api = self.configuration_api.get_token("ELEVENLABS_API_TOKEN")
            self.eleven = AsyncElevenLabs(api_key=self.eleven_labs_api)

            audio_stream = await self.eleven.generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2",
                stream=True
            )
            audio_data = b""
            async for chunk in audio_stream:
                audio_data += chunk

            audio_stream_io = io.BytesIO(audio_data)
            new_audio = await asyncio.to_thread(AudioSegment.from_file, audio_stream_io, format="mp3")
            self.audio_cache += new_audio
            await self.play_audio(new_audio)
        except Exception as e:
            logger.error(f"Error: {e}")

    async def generate_speech_with_elevenlabs_sow_system(self, text, voice_id):
        try:
            self.eleven_labs_api = self.configuration_api.get_token("ELEVENLABS_API_TOKEN")
            self.eleven = AsyncElevenLabs(api_key=self.eleven_labs_api)

            audio_stream = await self.eleven.generate(
                text=text,
                voice=voice_id,
                model="eleven_multilingual_v2",
                stream=True
            )
            audio_data = b""
            async for chunk in audio_stream:
                audio_data += chunk

            audio_stream_io = io.BytesIO(audio_data)
            new_audio = await asyncio.to_thread(AudioSegment.from_file, audio_stream_io, format="mp3")

            unique_id = uuid.uuid4().hex
            output_file = os.path.join(self.output_dir, f"output_{unique_id}.wav")
            await asyncio.to_thread(new_audio.export, output_file, format="wav")
            return output_file
        except Exception as e:
            logger.error(f"ElevenLabs Error: {e}")
            return None

    async def play_audio(self, audio):
        try:
            samples = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            sd.default.device = self.device_index
            await asyncio.to_thread(sd.play, samples, samplerate=sample_rate)
            await asyncio.to_thread(sd.wait)
        except Exception as e:
            logger.error(f"Error: {e}")

    def clear_audio_cache(self):
        self.audio_cache = AudioSegment.empty()


class XTTSv2_SOW_System:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = None
        self.tts_loaded = False
        self.rvc = None
        self.rvc_loaded = False

    def _load_tts_sync(self):
        if not self.tts_loaded:
            try:
                self.tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=True).to(self.device)
                self.tts_loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to load TTS model: {e}")

    async def _load_rvc(self, f0up_key, index_rate, protect):
        if not self.rvc_loaded:
            logger.info("Loading RVC model...")
            try:
                def _init():
                    return RVCInference(
                        models_dir="assets/rvc_models",
                        device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                        f0up_key=f0up_key, index_rate=index_rate, protect=protect
                    )
                self.rvc = await asyncio.to_thread(_init)
                self.rvc_loaded = True
                logger.info("RVC model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading RVC model: {e}")
                raise RuntimeError("Failed to load RVC model.")

    async def generate_speech_with_xttsv2_sow_system(self, text=None, language=None, character_name=None):
        await asyncio.to_thread(self._load_tts_sync)

        configuration_data = self.configuration_characters.load_configuration()
        char_config = configuration_data["character_list"][character_name]

        xttsv2_voice_type = char_config["voice_type"]
        xttsv2_rvc_enabled = char_config["rvc_enabled"]
        xttsv2_rvc_file = char_config["rvc_file"]

        speaker_wav_map = {
            "Female Calm": "app/voices/calm_female.wav",
            "Female": "app/voices/female.wav",
            "Male": "app/voices/male.wav"
        }
        speaker_wav = speaker_wav_map.get(xttsv2_voice_type)
        if not speaker_wav:
            raise ValueError(f"Unknown voice type: {xttsv2_voice_type}")

        os.makedirs("app/voices/xttsv2_audio", exist_ok=True)
        unique_id = uuid.uuid4().hex
        base_output_file = f"app/voices/xttsv2_audio/output_{unique_id}.wav"

        await asyncio.to_thread(
            self.tts.tts_to_file,
            text=text,
            speaker_wav=speaker_wav,
            language=language,
            file_path=base_output_file
        )

        if xttsv2_rvc_enabled and xttsv2_rvc_file:
            f0up_key   = char_config.get("rvc_f0up_key",   0)
            index_rate = char_config.get("rvc_index_rate", 0.75)
            protect    = char_config.get("rvc_protect",    0.5)

            await self._load_rvc(f0up_key, index_rate, protect)

            model_name = os.path.splitext(os.path.basename(xttsv2_rvc_file))[0]
            rvc_output_file = f"app/voices/xttsv2_audio/output_rvc_{unique_id}.wav"

            rvc_params = (model_name, f0up_key, index_rate, protect)
            if getattr(self, "_current_rvc_params", None) != rvc_params:
                await asyncio.to_thread(self.rvc.load_model, model_name)
                self._current_rvc_params = rvc_params

            await asyncio.to_thread(self.rvc.infer_file, base_output_file, rvc_output_file)

            fix_rvc_sample_rate(rvc_output_file, base_output_file)

            try:
                await asyncio.to_thread(os.remove, base_output_file)
            except OSError as e:
                logger.warning(f"Could not remove temp file {base_output_file}: {e}")

            return rvc_output_file

        return base_output_file


class EdgeTTS:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.rvc = None
        self.rvc_loaded = False
        self.output_dir = "app/voices/edge_tts_audio"
        os.makedirs(self.output_dir, exist_ok=True)
        self.device_index = self.configuration_settings.get_main_setting("output_device_real_index")

    async def _load_rvc(self, f0up_key, index_rate, protect):
        if not self.rvc_loaded:
            logger.info("Loading RVC model...")
            try:
                def _init():
                    return RVCInference(
                        models_dir="assets/rvc_models",
                        device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                        f0up_key=f0up_key, index_rate=index_rate, protect=protect
                    )
                self.rvc = await asyncio.to_thread(_init)
                self.rvc_loaded = True
                logger.info("RVC model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading RVC model: {e}")
                raise RuntimeError("Failed to load RVC model.")

    async def _convert_mp3_to_wav(self, mp3_file, wav_file):
        def _convert():
            audio = AudioSegment.from_mp3(mp3_file)
            audio.export(wav_file, format="wav")
        await asyncio.to_thread(_convert)

    async def _generate_base(self, text, character_name):
        configuration_data = self.configuration_characters.load_configuration()
        char_config = configuration_data["character_list"][character_name]

        voice_type = char_config["voice_type"]
        rvc_enabled = char_config["rvc_enabled"]
        rvc_file = char_config["rvc_file"]

        unique_id = uuid.uuid4().hex
        audio_file = os.path.join(self.output_dir, f"output_{unique_id}.mp3")
        wav_file = os.path.join(self.output_dir, f"output_{unique_id}.wav")

        try:
            communicate = edge_tts.Communicate(text, voice_type)
            await communicate.save(audio_file)
        except Exception as e:
            logger.error(f"Error when generating EdgeTTS audio: {e}")
            return None

        try:
            await self._convert_mp3_to_wav(audio_file, wav_file)
            await asyncio.to_thread(os.remove, audio_file)
        except Exception as e:
            logger.error(f"Error when converting MP3 to WAV: {e}")
            return None

        if rvc_enabled and rvc_file:
            f0up_key   = char_config.get("rvc_f0up_key",   0)
            index_rate = char_config.get("rvc_index_rate", 0.75)
            protect    = char_config.get("rvc_protect",    0.5)

            await self._load_rvc(f0up_key, index_rate, protect)
            model_name = os.path.splitext(os.path.basename(rvc_file))[0]
            rvc_output_file = os.path.join(self.output_dir, f"output_rvc_{unique_id}.wav")

            rvc_params = (model_name, f0up_key, index_rate, protect)
            if getattr(self, "_current_rvc_params", None) != rvc_params:
                await asyncio.to_thread(self.rvc.load_model, model_name)
                self._current_rvc_params = rvc_params

            await asyncio.to_thread(self.rvc.infer_file, wav_file, rvc_output_file)

            fix_rvc_sample_rate(rvc_output_file, wav_file)

            try:
                await asyncio.to_thread(os.remove, wav_file)
            except OSError as e:
                logger.warning(f"Could not remove temp file {wav_file}: {e}")

            return rvc_output_file

        return wav_file

    async def generate_speech_with_edge_tts(self, text, character_name):
        output_file = await self._generate_base(text, character_name)
        if output_file:
            await self.play_audio(file_path=output_file)

    async def generate_speech_with_edge_tts_sow_system(self, text, character_name):
        return await self._generate_base(text, character_name)

    async def play_audio(self, file_path):
        def _play():
            try:
                data, samplerate = sf.read(file_path, dtype='float32')
                sd.default.device = self.device_index
                sd.play(data, samplerate)
                sd.wait()
            except Exception as e:
                logger.error(f"Error: {e}")
        await asyncio.to_thread(_play)

    def stop_audio(self):
        sd.stop()


class KokoroTTS_SOW_System:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.tts_loaded = False
        self.rvc = None
        self.rvc_loaded = False

    async def _load_tts(self):
        if not self.tts_loaded:
            try:
                def _init():
                    return KPipeline(lang_code='a')
                self.pipeline = await asyncio.to_thread(_init)
                self.tts_loaded = True
            except Exception as e:
                raise RuntimeError(f"Failed to load TTS model: {e}")

    async def _load_rvc(self, f0up_key, index_rate, protect):
        if not self.rvc_loaded:
            logger.info("Loading RVC model...")
            try:
                def _init():
                    return RVCInference(
                        models_dir="assets/rvc_models",
                        device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                        f0up_key=f0up_key, index_rate=index_rate, protect=protect
                    )
                self.rvc = await asyncio.to_thread(_init)
                self.rvc_loaded = True
                logger.info("RVC model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading RVC model: {e}")
                raise RuntimeError("Failed to load RVC model.")

    async def generate_speech_with_kokoro(self, text, character_name):
        await self._load_tts()

        configuration_data = self.configuration_characters.load_configuration()
        char_config = configuration_data["character_list"][character_name]
        # A character set up for another engine (e.g. Qwen cloning) has an
        # empty voice_type; an empty name made Kokoro fetch "voices/.pt"
        # (HTTP 404) every time this fallback chain was walked.
        kokoro_voice_name = char_config.get("voice_type") or "af_heart"
        kokoro_rvc_enabled = char_config.get("rvc_enabled")
        kokoro_rvc_file = char_config.get("rvc_file")

        def _generate():
            generator = self.pipeline(text, voice=kokoro_voice_name)
            chunks = []
            for _, _, audio in generator:
                chunks.append(audio)
            return chunks

        all_audio = await asyncio.to_thread(_generate)

        if not all_audio:
            logger.warning("Kokoro generated empty audio for this chunk. Skipping.")
            return None

        full_audio = np.concatenate(all_audio)
        os.makedirs("app/voices/kokoro_audio", exist_ok=True)
        unique_id = uuid.uuid4().hex
        base_output_file = f"app/voices/kokoro_audio/kokoro_output_{unique_id}.wav"
        await asyncio.to_thread(sf.write, base_output_file, full_audio, 24000)

        if kokoro_rvc_enabled and kokoro_rvc_file:
            f0up_key   = char_config.get("rvc_f0up_key",   0)
            index_rate = char_config.get("rvc_index_rate", 0.75)
            protect    = char_config.get("rvc_protect",    0.5)

            await self._load_rvc(f0up_key, index_rate, protect)
            model_name = os.path.splitext(os.path.basename(kokoro_rvc_file))[0]
            rvc_output_file = f"app/voices/kokoro_audio/output_rvc_{unique_id}.wav"

            rvc_params = (model_name, f0up_key, index_rate, protect)
            if getattr(self, "_current_rvc_params", None) != rvc_params:
                await asyncio.to_thread(self.rvc.load_model, model_name)
                self._current_rvc_params = rvc_params

            await asyncio.to_thread(self.rvc.infer_file, base_output_file, rvc_output_file)

            fix_rvc_sample_rate(rvc_output_file, base_output_file)

            try:
                await asyncio.to_thread(os.remove, base_output_file)
            except OSError as e:
                logger.warning(f"Could not remove temp file {base_output_file}: {e}")

            return rvc_output_file

        return base_output_file


class SileroTTS_SOW_System:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tts_loaded = False
        self.rvc = None
        self.rvc_loaded = False

    async def _load_tts(self):
        if not self.tts_loaded:
            try:
                self.model, _ = await asyncio.to_thread(
                    torch.hub.load,
                    repo_or_dir='snakers4/silero-models',
                    model='silero_tts',
                    language='ru',
                    speaker='v5_3_ru',
                    trust_repo=True,
                    force_reload=False,
                    verbose=True
                )
                self.model.to(self.device)
                self.tts_loaded = True
                logger.info("Silero TTS loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Silero TTS: {e}")

    async def _load_rvc(self, f0up_key, index_rate, protect):
        if not self.rvc_loaded:
            logger.info("Loading RVC model...")
            try:
                def _init():
                    return RVCInference(
                        models_dir="assets/rvc_models",
                        device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                        f0up_key=f0up_key, index_rate=index_rate, protect=protect
                    )
                self.rvc = await asyncio.to_thread(_init)
                self.rvc_loaded = True
                logger.info("RVC model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading RVC model: {e}")
                raise RuntimeError("Failed to load RVC model.")

    async def generate_speech_with_silero(self, text, character_name):
        await self._load_tts()

        configuration_data = self.configuration_characters.load_configuration()
        char_config = configuration_data["character_list"][character_name]

        silero_voice = char_config.get("voice_type", "xenia")
        silero_rvc_enabled = char_config.get("rvc_enabled", False)
        silero_rvc_file = char_config.get("rvc_file")

        os.makedirs("app/voices/silero_audio", exist_ok=True)
        unique_id = uuid.uuid4().hex
        base_output_file = f"app/voices/silero_audio/silero_output_{unique_id}.wav"

        audio = await asyncio.to_thread(
            self.model.apply_tts,
            text=text,
            speaker=silero_voice,
            sample_rate=48000
        )
        await asyncio.to_thread(sf.write, base_output_file, audio.cpu().numpy(), 48000)

        if silero_rvc_enabled and silero_rvc_file:
            f0up_key   = char_config.get("rvc_f0up_key",   0)
            index_rate = char_config.get("rvc_index_rate", 0.75)
            protect    = char_config.get("rvc_protect",    0.5)

            await self._load_rvc(f0up_key, index_rate, protect)
            model_name = os.path.splitext(os.path.basename(silero_rvc_file))[0]
            rvc_output_file = f"app/voices/silero_audio/output_rvc_{unique_id}.wav"

            rvc_params = (model_name, f0up_key, index_rate, protect)
            if getattr(self, "_current_rvc_params", None) != rvc_params:
                await asyncio.to_thread(self.rvc.load_model, model_name)
                self._current_rvc_params = rvc_params
            
            await asyncio.to_thread(self.rvc.infer_file, base_output_file, rvc_output_file)

            fix_rvc_sample_rate(rvc_output_file, base_output_file)

            try:
                await asyncio.to_thread(os.remove, base_output_file)
            except OSError as e:
                logger.warning(f"Could not remove temp file {base_output_file}: {e}")

            return rvc_output_file

        return base_output_file

class Qwen3TTS_SOW_System:
    def __init__(self):
        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.tts_loaded = False
        self.rvc = None
        self.rvc_loaded = False

        # Caches: characters.json used to be re-parsed on every sentence (and
        # racing writers occasionally produced truncated JSON, silently
        # dropping the engine to the fallback chain); the voice-clone prompt
        # re-encoded the reference audio (speech tokenizer + speaker embedding
        # extraction) on every call — seconds of pure overhead per sentence.
        self._char_cfg_cache = None
        self._char_cfg_key = None
        self._clone_prompt_cache = {}
        self._tts_executor = None

    def _get_tts_executor(self):
        """Dedicated single-thread executor for blocking TTS work.

        The event loop's default executor was observed dying mid-session
        ('cannot schedule new futures after shutdown', right after a CUDA OOM
        in a worker thread), which killed every subsequent batch attempt.
        Owning the pool makes generation self-healing: _run_blocking recreates
        it if that ever happens again.
        """
        if self._tts_executor is None:
            self._tts_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="qwen-tts"
            )
        return self._tts_executor

    async def _run_blocking(self, fn, *args):
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(self._get_tts_executor(), lambda: fn(*args))
        except RuntimeError as e:
            if "after shutdown" not in str(e):
                raise
            self._tts_executor = None
            return await loop.run_in_executor(self._get_tts_executor(), lambda: fn(*args))

    def _get_char_config(self, character_name: str) -> dict:
        """characters.json read, cached per file mtime, with a retry loop.

        Concurrent config writers can momentarily expose truncated JSON
        ('Unterminated string'); retrying instead of failing keeps the
        sentence on the Qwen engine.
        """
        try:
            mtime = os.path.getmtime(self.configuration_characters.characters_path)
        except OSError:
            mtime = None
        key = (character_name, mtime)
        if self._char_cfg_key == key and self._char_cfg_cache is not None:
            return self._char_cfg_cache

        last_err = None
        for _ in range(3):
            try:
                data = self.configuration_characters.load_configuration()
                char_config = data["character_list"].get(character_name, {})
                self._char_cfg_cache = char_config
                self._char_cfg_key = key
                return char_config
            except Exception as e:
                last_err = e
                time.sleep(0.15)
        logger.warning(f"Qwen3-TTS: characters.json read failed ({last_err}); using cached config")
        return self._char_cfg_cache or {}

    def _get_voice_clone_prompt(self, ref_path: str, ref_text: str):
        """Voice-clone prompt cached per reference file.

        create_voice_clone_prompt() re-runs the speech tokenizer and speaker
        embedding extraction over the reference wav on every call; the result
        only depends on the audio file itself.
        """
        try:
            mtime = os.path.getmtime(ref_path)
        except OSError:
            mtime = None
        key = (ref_path, bool(ref_text), mtime)
        cached = self._clone_prompt_cache.get(key)
        if cached is not None:
            return cached

        if not ref_text:
            logger.warning(
                f"Qwen3-TTS: no 'qwen_cloning_ref_text' set for '{ref_path}'; "
                f"using x_vector_only_mode (cloning quality may be reduced)."
            )
        items = self.model.create_voice_clone_prompt(
            ref_audio=ref_path,
            ref_text=ref_text or None,
            x_vector_only_mode=not bool(ref_text),
        )
        if len(self._clone_prompt_cache) > 4:
            self._clone_prompt_cache.clear()
        self._clone_prompt_cache[key] = items
        return items

    async def _load_tts(self, character_name: str):
        if not self.tts_loaded:
            try:
                char_config = self._get_char_config(character_name)

                model_size = char_config.get("qwen_model_size", "1.7B")
                qwen_mode = char_config.get("qwen_mode", "presets")
                qwen_device = char_config.get("qwen_device", "cpu")

                if qwen_device == "cuda" and torch.cuda.is_available():
                    self.device = "cuda"
                    target_dtype = torch.bfloat16
                    target_device_map = "auto"
                else:
                    self.device = "cpu"
                    target_dtype = torch.float32
                    target_device_map = None

                if qwen_mode == "prompt":
                    variant = "VoiceDesign"
                elif qwen_mode == "cloning":
                    variant = "Base"
                else:
                    variant = "CustomVoice"

                if variant == "VoiceDesign" and model_size != "1.7B":
                    logger.warning(
                        f"Qwen3-TTS: VoiceDesign is only available as 1.7B, "
                        f"got model_size='{model_size}'. Falling back to 1.7B."
                    )
                    model_size = "1.7B"

                model_name = f"Qwen/Qwen3-TTS-12Hz-{model_size}-{variant}"

                def _init():
                    if target_device_map == "auto":
                        # device_map="auto" silently spreads layers by leftover
                        # VRAM and parks the overflow on CPU — generation slows
                        # down several-fold. Pin the whole model to the GPU
                        # first; only fall back to "auto" if it doesn't fit.
                        try:
                            return Qwen3TTSModel.from_pretrained(
                                model_name,
                                dtype=target_dtype,
                                device_map={"": "cuda:0"},
                                attn_implementation="sdpa",
                            )
                        except Exception as e:
                            logger.warning(
                                f"Qwen3-TTS: full-GPU pin failed ({e}); "
                                "falling back to device_map='auto'."
                            )
                    model = Qwen3TTSModel.from_pretrained(
                        model_name,
                        dtype=target_dtype,
                        device_map=target_device_map,
                        attn_implementation="sdpa",
                    )
                    
                    if target_dtype == torch.float32:
                        def clean_and_cast_to_float32(obj, visited=None):
                            if visited is None:
                                visited = set()
                            
                            obj_id = id(obj)
                            if obj_id in visited:
                                return
                            visited.add(obj_id)

                            if isinstance(obj, torch.nn.Module):
                                if hasattr(obj, "_hf_hook"):
                                    try:
                                        delattr(obj, "_hf_hook")
                                    except Exception:
                                        pass
                                try:
                                    obj.to(torch.float32)
                                except Exception:
                                    pass
                                for child in obj.children():
                                    clean_and_cast_to_float32(child, visited)
                            
                            elif hasattr(obj, "__dict__"):
                                for attr_name, attr_val in list(obj.__dict__.items()):
                                    if attr_name.startswith("__"):
                                        continue
                                    clean_and_cast_to_float32(attr_val, visited)
                            
                            elif isinstance(obj, (list, tuple)):
                                for item in obj:
                                    clean_and_cast_to_float32(item, visited)
                                    
                            elif isinstance(obj, dict):
                                for value in obj.values():
                                    clean_and_cast_to_float32(value, visited)

                        clean_and_cast_to_float32(model)
                            
                    return model

                self.model = await self._run_blocking(_init)
                self.tts_loaded = True
                
                logger.info(f"Qwen3-TTS {model_size}-{variant} loaded on {self.device.upper()} for '{character_name}'")

            except Exception as e:
                if "out of memory" in str(e).lower():
                    logger.error(
                        "Qwen3-TTS: CUDA out of memory while loading. llama-server is keeping too much "
                        "VRAM for the TTS model to fit next to it. Lower 'GPU layers' in Options "
                        "(e.g. 40) or switch this character to the 0.6B TTS model."
                    )
                logger.error(f"Failed to load Qwen3-TTS for {character_name}: {e}")
                raise RuntimeError(f"Failed to load Qwen3-TTS: {e}")

    async def _load_rvc(self, f0up_key, index_rate, protect):
        if not self.rvc_loaded:
            logger.info("Loading RVC model for Qwen 3...")
            try:
                def _init():
                    return RVCInference(
                        models_dir="assets/rvc_models",
                        device="cuda:0" if torch.cuda.is_available() else "cpu:0",
                        f0up_key=f0up_key, index_rate=index_rate, protect=protect
                    )
                self.rvc = await self._run_blocking(_init)
                self.rvc_loaded = True
                logger.info("RVC model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading RVC model: {e}")
                raise RuntimeError("Failed to load RVC model.")

    async def generate_speech_with_qwen3(self, text: str, character_name: str):
        await self._load_tts(character_name)

        char_config = self._get_char_config(character_name)

        qwen_mode = char_config.get("qwen_mode", "presets")
        voice_type = char_config.get("voice_type", "Serena")
        qwen_prompt = char_config.get("qwen_prompt", "")
        qwen_instruct = char_config.get("qwen_style_instruct", "")
        qwen_ref_path = char_config.get("qwen_cloning_ref_path", "")
        qwen_ref_text = char_config.get("qwen_cloning_ref_text", "")
        language = char_config.get("qwen_language", "English")

        qwen_rvc_enabled = char_config.get("rvc_enabled", False)
        qwen_rvc_file = char_config.get("rvc_file")

        os.makedirs("app/voices/qwen_audio", exist_ok=True)
        unique_id = uuid.uuid4().hex
        base_output_file = f"app/voices/qwen_audio/qwen_output_{unique_id}.wav"

        def _generate():
            try:
                generation_params = {
                    "text": text,
                    "language": language,
                    "max_new_tokens": 1500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }

                with torch.no_grad():
                    if self.device == "cpu":
                        autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)
                    else:
                        autocast_ctx = contextlib.nullcontext()

                    with autocast_ctx:
                        if qwen_mode == "cloning" and qwen_ref_path and os.path.exists(qwen_ref_path):
                            # Voice Cloning — the prompt (speaker embedding /
                            # ref codes) is cached per reference file instead
                            # of being rebuilt for every sentence.
                            prompt_items = self._get_voice_clone_prompt(qwen_ref_path, qwen_ref_text)
                            wavs, sr = self.model.generate_voice_clone(
                                voice_clone_prompt=prompt_items,
                                **generation_params
                            )

                        elif qwen_mode == "prompt" and qwen_prompt:
                            # Voice Design
                            wavs, sr = self.model.generate_voice_design(
                                instruct=qwen_prompt,
                                **generation_params
                            )

                        else:
                            # Preset Voices (CustomVoice)
                            wavs, sr = self.model.generate_custom_voice(
                                speaker=voice_type, instruct=qwen_instruct,
                                **generation_params
                            )

                audio_data = wavs[0]
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().cpu().to(torch.float32).numpy()
                elif isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.astype(np.float32)

                sf.write(base_output_file, audio_data, sr)
                return base_output_file

            except Exception as e:
                logger.error(f"Qwen3 TTS generation error: {e}")
                raise

        await self._run_blocking(_generate)

        if qwen_rvc_enabled and qwen_rvc_file:
            f0up_key = char_config.get("rvc_f0up_key", 0)
            index_rate = char_config.get("rvc_index_rate", 0.75)
            protect = char_config.get("rvc_protect", 0.5)

            await self._load_rvc(f0up_key, index_rate, protect)

            model_name = os.path.splitext(os.path.basename(qwen_rvc_file))[0]
            rvc_output_file = f"app/voices/qwen_audio/output_rvc_{unique_id}.wav"

            rvc_params = (model_name, f0up_key, index_rate, protect)
            if getattr(self, "_current_rvc_params", None) != rvc_params:
                await asyncio.to_thread(self.rvc.load_model, model_name)
                self._current_rvc_params = rvc_params

            await asyncio.to_thread(self.rvc.infer_file, base_output_file, rvc_output_file)

            fix_rvc_sample_rate(rvc_output_file, base_output_file)

            try:
                os.remove(base_output_file)
            except OSError:
                pass

            return rvc_output_file

        return base_output_file

    async def generate_speech_batch_with_qwen3(self, texts, character_name: str):
        """Batched generation: several sentences synthesized in ONE model pass.

        Qwen3-TTS is latency-bound per token step (~0.1 s/step on a mid GPU),
        so a single short sentence takes longer to generate than to play
        (RTF ~1.3 measured on RTX 3060) and per-sentence generation can never
        keep speech continuous. A batch of N sentences runs in the same number
        of autoregressive steps as its longest member, i.e. N× the audio for
        roughly the same wall time.

        Returns a list of wav paths matching `texts`, or None when batching is
        not applicable (single text, RVC post-processing, generation error) —
        the caller then falls back to the per-sentence engine chain.
        """
        if not texts or len(texts) < 2:
            return None

        await self._load_tts(character_name)
        char_config = self._get_char_config(character_name)

        qwen_mode = char_config.get("qwen_mode", "presets")
        voice_type = char_config.get("voice_type", "Serena")
        qwen_prompt = char_config.get("qwen_prompt", "")
        qwen_ref_path = char_config.get("qwen_cloning_ref_path", "")
        qwen_ref_text = char_config.get("qwen_cloning_ref_text", "")
        language = char_config.get("qwen_language", "English")

        # RVC post-processing is per-file; keep those characters on the
        # per-sentence path.
        if char_config.get("rvc_enabled", False):
            return None

        os.makedirs("app/voices/qwen_audio", exist_ok=True)

        def _generate():
            gen_params = {
                "max_new_tokens": 1500,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            with torch.no_grad():
                if qwen_mode == "cloning" and qwen_ref_path and os.path.exists(qwen_ref_path):
                    prompt_items = self._get_voice_clone_prompt(qwen_ref_path, qwen_ref_text)
                    wavs, sr = self.model.generate_voice_clone(
                        text=texts, language=language,
                        voice_clone_prompt=prompt_items * len(texts),
                        **gen_params
                    )
                elif qwen_mode == "prompt" and qwen_prompt:
                    wavs, sr = self.model.generate_voice_design(
                        text=texts, instruct=qwen_prompt, **gen_params
                    )
                else:
                    wavs, sr = self.model.generate_custom_voice(
                        text=texts, speaker=voice_type,
                        instruct=char_config.get("qwen_style_instruct", ""),
                        **gen_params
                    )

            files = []
            for audio in wavs:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().to(torch.float32).numpy()
                elif isinstance(audio, np.ndarray):
                    audio = audio.astype(np.float32)
                output_file = f"app/voices/qwen_audio/qwen_output_{uuid.uuid4().hex}.wav"
                sf.write(output_file, audio, sr)
                files.append(output_file)
            return files

        try:
            files = await self._run_blocking(_generate)
        except torch.cuda.OutOfMemoryError:
            # llama-server transiently holds its compute buffers while it
            # processes the prompt; releasing our allocator cache and retrying
            # usually fits. If not, the per-sentence fallback needs far less.
            logger.warning("Qwen3-TTS: batch OOM — releasing allocator cache and retrying once")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                files = await self._run_blocking(_generate)
            except Exception as e:
                logger.warning(f"Qwen3-TTS: batch OOM on retry ({e}); falling back per sentence")
                return None
        except Exception as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "Qwen3-TTS: CUDA out of memory during batch generation. llama-server + TTS no longer "
                    "fit together — lower 'GPU layers' in Options (e.g. 40) or use the 0.6B TTS model."
                )
            logger.warning(f"Qwen3-TTS: batch generation failed ({e}); falling back per sentence")
            return None

        if not files or len(files) != len(texts):
            return None
        return files

class AudioPlaybackWorker(QThread):
    queue_empty_signal = pyqtSignal()
    lipsync_signal = pyqtSignal(float)

    def __init__(self, device_index):
        super().__init__()
        self.queue = queue.Queue()
        self.is_running = True
        self.device_index = device_index
        self.interrupt_flag = False

    def add_audio_file(self, file_path, persist=False):
        self.interrupt_flag = False
        self.queue.put((file_path, persist))

    def _device_output_rate(self) -> int:
        """Declared default rate of the configured output device (fallback 44100)."""
        try:
            index = self.device_index
            if index is None:
                index = sd.query_devices(kind='output')['index']
            rate = int(sd.query_devices(index)['default_samplerate'])
            return rate if rate > 0 else 44100
        except Exception:
            return 44100

    @staticmethod
    def _resample(data, sr_in: int, sr_out: int):
        """Linear-interpolation resample; keeps float32 and channel layout."""
        if not sr_in or sr_in == sr_out:
            return data
        n_out = int(round(data.shape[0] * sr_out / sr_in))
        if n_out < 1:
            return data
        x_new = np.linspace(0.0, data.shape[0] - 1, num=n_out)
        x_old = np.arange(data.shape[0])
        if data.ndim == 1:
            return np.interp(x_new, x_old, data).astype(np.float32)
        return np.stack(
            [np.interp(x_new, x_old, data[:, c]) for c in range(data.shape[1])],
            axis=1,
        ).astype(np.float32)

    def clear_queue(self):
        self.interrupt_flag = True
        with self.queue.mutex:
            for item in list(self.queue.queue):
                file_path, persist = item if isinstance(item, tuple) else (item, False)
                if persist:
                    continue
                try:
                    if file_path and os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
            self.queue.queue.clear()

    def run(self):
        logger.info("Audio Playback Worker Started")
        while self.is_running:
            try:
                item = self.queue.get(timeout=0.1)

                if item is None:
                    self.queue.task_done()
                    continue

                file_path, persist = item if isinstance(item, tuple) else (item, False)

                if self.interrupt_flag:
                    if not persist:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception:
                            pass
                    self.queue.task_done()
                    continue

                try:
                    data, samplerate = sf.read(file_path, dtype='float32')
                    sd.default.device = self.device_index

                    # Neural TTS engines emit 24 kHz while raw ALSA hw devices
                    # only accept their declared rate (44.1/48 kHz) — PortAudio
                    # answers paInvalidSampleRate (-9997). Resample to the
                    # device's rate and, if the stream still refuses (channel
                    # count restrictions), retry mono->stereo, then the default
                    # output device.
                    try:
                        sd.play(data, samplerate)
                    except sd.PortAudioError:
                        dev_rate = self._device_output_rate()
                        data = self._resample(data, samplerate, dev_rate)
                        samplerate = dev_rate
                        try:
                            sd.play(data, samplerate)
                        except sd.PortAudioError:
                            if data.ndim == 1:
                                data = np.stack([data, data], axis=1)
                            try:
                                sd.play(data, samplerate)
                            except sd.PortAudioError:
                                sd.default.device = None
                                sd.play(data, samplerate)

                    duration = len(data) / samplerate
                    chunk_size = int(samplerate * 0.05)
                    slept = 0

                    while slept < duration:
                        if self.interrupt_flag:
                            sd.stop()
                            self.lipsync_signal.emit(0.0)
                            break

                        start_idx = int(slept * samplerate)
                        end_idx = min(start_idx + chunk_size, len(data))
                        current_chunk = data[start_idx:end_idx]

                        if len(current_chunk) > 0:
                            rms = np.sqrt(np.mean(current_chunk ** 2))
                            mouth_open = min(rms * 5.0, 1.0)
                            self.lipsync_signal.emit(float(mouth_open))

                        time.sleep(0.05)
                        slept += 0.05

                    self.lipsync_signal.emit(0.0)

                except Exception as e:
                    logger.error(f"Playback error: {e}")

                finally:
                    if not persist:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                        except Exception:
                            pass

                self.queue.task_done()

                if self.queue.empty() and not self.interrupt_flag:
                    self.queue_empty_signal.emit()

            except queue.Empty:
                continue

    def stop(self):
        self.is_running = False
        self.interrupt_flag = True
        try:
            sd.stop()
        except Exception:
            pass
        self.queue.put(None)
        self.quit()
        self.wait()


class TTSWorker(QThread):
    audio_ready_signal = pyqtSignal(str)
    segment_saved_signal = pyqtSignal(str, str)

    # Qwen-3 TTS: max sentences synthesized in one model pass. The model is
    # latency-bound per token step, so a batch costs about as much as its
    # longest sentence while producing the audio of all of them.
    QWEN_TTS_BATCH_MAX = 6
    # The worker briefly waits for the next streamed sentences to join the
    # batch: a group of 1 generates SLOWER than realtime (RTF ~1.3), a group
    # of 3+ is what keeps playback continuous.
    QWEN_TTS_GROW_TARGET = 3
    QWEN_TTS_GROW_WAIT_SEC = 2.0

    def __init__(self, tts_method, character_name, voice_id=None, language="en"):
        super().__init__()
        self.queue = queue.Queue()
        self.is_running = True
        self.discard_current = False
        # True while the LLM reply is still streaming (more sentences coming);
        # finish_stream() flips it off so the batch grow-phase stops waiting.
        self._stream_active = False

        self.tts_method = tts_method
        self.character_name = character_name
        self.voice_id = voice_id
        self.language = language

        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device_index = self.configuration_settings.get_main_setting("output_device_real_index")

        self.tts_mode = self.configuration_settings.get_main_setting("tts_voicing_mode") or 0
        self.tts_custom_regex = self.configuration_settings.get_main_setting("tts_custom_regex") or ""

        self._in_tts_quote = False
        self._in_asterisk = False

        self.xtts = XTTSv2_SOW_System()
        self.edge = EdgeTTS()
        self.kokoro = KokoroTTS_SOW_System()
        self.silero = SileroTTS_SOW_System()
        self.qwen = Qwen3TTS_SOW_System()
        self.eleven = ElevenLabs()
        self.comfy_tts = None
        self._fallback_warned = set()

        self.playback_worker = AudioPlaybackWorker(self.device_index)
        self.playback_worker.start()

    def _get_comfy_tts(self):
        if self.comfy_tts is None:
            from app.utils.providers.comfy_tts import ComfyUITTS
            from app.utils.media.comfy_client import ComfyUIClient
            self.comfy_tts = ComfyUITTS(ComfyUIClient())
        return self.comfy_tts

    def _generate_with_comfyui_fallback(self, loop, text) -> Optional[str]:
        """Legacy entry point: ComfyUI workflow, падение — на Kokoro."""
        from app.utils.providers.comfy_tts import configured_tts_chain
        return self._run_chain(configured_tts_chain(self.tts_method), loop, text)

    def _generate_with_engine_chain(self, loop, text) -> Optional[str]:
        """Phase-B: порядок попыток задаёт media.yaml:tts_engine_order.

        Выбор GUI (self.tts_method) всегда первый; остальные элементы
        списка — фолбэки на случай ошибки/пустого результата.
        """
        from app.utils.providers.comfy_tts import configured_tts_chain
        return self._run_chain(configured_tts_chain(self.tts_method), loop, text)

    def _run_chain(self, chain, loop, text) -> Optional[str]:
        from app.utils.providers.comfy_tts import run_fallback_chain
        return run_fallback_chain(
            chain,
            lambda method: self._dispatch_engine(method, loop, text),
            on_fallback=lambda cur, nxt, exc: log_tts_engine_fallback(self, cur, nxt, exc),
        )

    def _dispatch_engine(self, method, loop, text) -> Optional[str]:
        if method == "XTTSv2":
            return loop.run_until_complete(
                self.xtts.generate_speech_with_xttsv2_sow_system(text, self.language, self.character_name)
            )
        elif method == "Edge TTS":
            return loop.run_until_complete(
                self.edge.generate_speech_with_edge_tts_sow_system(text, self.character_name)
            )
        elif method == "Kokoro":
            return loop.run_until_complete(
                self.kokoro.generate_speech_with_kokoro(text, self.character_name)
            )
        elif method == "Silero":
            return loop.run_until_complete(
                self.silero.generate_speech_with_silero(text, self.character_name)
            )
        elif method == "Qwen-3 TTS":
            return loop.run_until_complete(
                self.qwen.generate_speech_with_qwen3(text, self.character_name)
            )
        elif method == "ElevenLabs":
            return loop.run_until_complete(
                self.eleven.generate_speech_with_elevenlabs_sow_system(text, self.voice_id)
            )
        elif method == "ComfyUI":
            return loop.run_until_complete(
                self._get_comfy_tts().generate(text, self.character_name)
            )
        logger.error(f"Unsupported TTS method: {method}")
        return None

    def add_text(self, text, message_id=None):
        if not text:
            return

        self._stream_active = True

        text = self.clean_text_for_speech(text)
        if not text:
            return

        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        max_chunk_len = 450

        def _enqueue(chunk_text):
            if chunk_text:
                self.queue.put((chunk_text, message_id))

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chunk_len:
                if current_chunk:
                    _enqueue(current_chunk)
                    current_chunk = ""

                sub_chunks = re.split(r'(?<=[,;])\s', sentence)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if len(sub) > max_chunk_len:
                        for i in range(0, len(sub), max_chunk_len):
                            _enqueue(sub[i:i+max_chunk_len])
                    else:
                        _enqueue(sub)
            else:
                if len(current_chunk) + len(sentence) + 1 > max_chunk_len:
                    _enqueue(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

        if current_chunk:
            _enqueue(current_chunk)

    def finish_stream(self):
        """The LLM reply finished streaming: no more sentences are coming, so
        the Qwen batch grow-phase stops waiting and starts immediately."""
        self._stream_active = False

    def _persist_segment_for_replay(self, source_file, message_id):
        if not source_file or not os.path.exists(source_file):
            return None

        segment_dir = os.path.join(
            os.getcwd(), "app", "data", ".soul", self.character_name, "tts_audio", message_id
        )
        os.makedirs(segment_dir, exist_ok=True)

        existing = [f for f in os.listdir(segment_dir) if f.lower().endswith((".wav", ".mp3"))]
        next_index = len(existing) + 1
        ext = os.path.splitext(source_file)[1] or ".wav"
        segment_filename = f"seg_{next_index:04d}{ext}"
        destination = os.path.join(segment_dir, segment_filename)

        shutil.copyfile(source_file, destination)

        return f"tts_audio/{message_id}/{segment_filename}"

    def clean_text_for_speech(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        mode = self.tts_mode
        regex = self.tts_custom_regex

        # --- Mode 0: Voice Everything ---
        if mode == 0:
            return re.sub(r"[*_~`#]", "", text).strip()

        # --- Mode 1: Voice Only Quotes ---
        elif mode == 1:
            result = []
            parts = re.split(r'(["“”«»])', text)
            for part in parts:
                if part in ('"', "“", "”", "«", "»"):
                    self._in_tts_quote = not self._in_tts_quote
                    if not self._in_tts_quote:
                        result.append(" ")
                    continue
                if self._in_tts_quote:
                    result.append(part)
            return re.sub(r"[*_~`#]", "", "".join(result)).strip()

        # --- Mode 2: Ignore Asterisks ---
        elif mode == 2:
            result = []
            parts = re.split(r"(\*|_)", text)
            for part in parts:
                if part in ("*", "_"):
                    self._in_asterisk = not self._in_asterisk
                    if not self._in_asterisk:
                        result.append(" ")
                    continue
                if not self._in_asterisk:
                    result.append(part)
            return re.sub(r"[*_~`#]", "", "".join(result)).strip()

        # --- Mode 3: Voice Outside Quotes ---
        elif mode == 3:
            result = []
            parts = re.split(r'(["“”«»])', text)
            for part in parts:
                if part in ('"', "“", "”", "«", "»"):
                    if part in ("“", "«"):
                        self._in_tts_quote = True
                    elif part in ("”", "»"):
                        self._in_tts_quote = False
                    else:
                        self._in_tts_quote = not self._in_tts_quote

                    if not self._in_tts_quote:
                        result.append(" ")
                    continue

                if not self._in_tts_quote:
                    result.append(part)

            return re.sub(r"[*_~`#]", "", "".join(result)).strip()

        # --- Mode 4: Custom Regex ---
        elif mode == 4:
            if not regex:
                return re.sub(r"[*_~`#]", "", text).strip()
            try:
                matches = re.findall(regex, text)
                if matches:
                    extracted = []
                    for m in matches:
                        if isinstance(m, tuple):
                            val = next((g for g in m if g), "")
                            extracted.append(val)
                        else:
                            extracted.append(m)
                    return " ".join(extracted).strip()
                return ""
            except re.error:
                return re.sub(r"[*_~`#]", "", text).strip()

        return text.strip()
    
    def clear_queue(self):
        with self.queue.mutex:
            self.queue.queue.clear()
        self.discard_current = True
        self._stream_active = False
        self._in_tts_quote = False
        self._in_asterisk = False
        if hasattr(self, 'playback_worker'):
            self.playback_worker.clear_queue()

    def _deliver_output(self, output_file, message_id):
        """Encode + persist + queue one finished wav for playback."""
        if not output_file or not os.path.exists(output_file):
            return
        try:
            import base64
            with open(output_file, "rb") as f:
                b64_audio = base64.b64encode(f.read()).decode("utf-8")
            self.audio_ready_signal.emit(b64_audio)
        except Exception as e:
            logger.error(f"Error encoding audio for web client: {e}")

        if message_id:
            try:
                relative_path = self._persist_segment_for_replay(output_file, message_id)
                if relative_path:
                    self.segment_saved_signal.emit(message_id, relative_path)
            except Exception as e:
                logger.error(f"Failed to persist TTS segment for replay: {e}")

        self.playback_worker.add_audio_file(output_file)

    def run(self):
        logger.info(f"TTS Worker Started ({self.tts_method})")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.is_running:
            try:
                item = self.queue.get(timeout=0.5)

                if item is None:
                    self.queue.task_done()
                    continue

                text, message_id = item if isinstance(item, tuple) else (item, None)

                if text:
                    self.discard_current = False

                    # Qwen-3 TTS: GROW PHASE — while the reply is still
                    # streaming, newly arrived sentences are folded into this
                    # batch instead of being generated one by one. Per-sentence
                    # generation is slower than realtime (RTF ~1.3 measured),
                    # so a lone first sentence is exactly what produced the
                    # silent gap after it: the follow-up batch outgrew the
                    # playback head start. A bounded wait (~2 s, target 3
                    # sentences) buys a group whose audio outlasts its own
                    # generation time. The moment the stream ends, generation
                    # starts immediately with whatever was collected.
                    batch = [(text, message_id)]
                    if self.tts_method == "Qwen-3 TTS":
                        grow_deadline = time.monotonic() + self.QWEN_TTS_GROW_WAIT_SEC
                        while (self.is_running
                               and not self.discard_current
                               and self._stream_active
                               and len(batch) < self.QWEN_TTS_BATCH_MAX
                               and len(batch) < self.QWEN_TTS_GROW_TARGET):
                            wait = grow_deadline - time.monotonic()
                            if wait <= 0:
                                break
                            try:
                                nxt = self.queue.get(timeout=min(0.2, wait))
                            except queue.Empty:
                                continue
                            if nxt is None:
                                self.queue.put(None)
                                break
                            n_text, n_mid = nxt if isinstance(nxt, tuple) else (nxt, None)
                            if n_text:
                                batch.append((n_text, n_mid))

                        # Final non-blocking drain: don't leave queued
                        # sentences for a later, smaller batch.
                        while len(batch) < self.QWEN_TTS_BATCH_MAX:
                            try:
                                nxt = self.queue.get_nowait()
                            except queue.Empty:
                                break
                            if nxt is None:
                                self.queue.put(None)
                                break
                            n_text, n_mid = nxt if isinstance(nxt, tuple) else (nxt, None)
                            if n_text:
                                batch.append((n_text, n_mid))

                    if len(batch) > 1:
                        logger.info(f"TTS batch: {len(batch)} sentences in one pass")
                        batch_files = None
                        try:
                            batch_files = loop.run_until_complete(
                                self.qwen.generate_speech_batch_with_qwen3(
                                    [b_text for b_text, _ in batch], self.character_name
                                )
                            )
                        except Exception as e:
                            logger.warning(f"TTS batch generation failed: {e}")

                        if batch_files and len(batch_files) == len(batch):
                            if self.discard_current:
                                logger.info("TTS batch finished, but was interrupted. Discarding audio.")
                                for output_file in batch_files:
                                    if output_file and os.path.exists(output_file):
                                        try:
                                            os.remove(output_file)
                                        except Exception:
                                            pass
                                for _ in batch:
                                    self.queue.task_done()
                                continue

                            for (b_text, b_mid), output_file in zip(batch, batch_files):
                                logger.info(f"Batch audio ready: {b_text[:30]}...")
                                self._deliver_output(output_file, b_mid)
                                self.queue.task_done()
                            continue
                        # Batch unavailable (RVC, single-mode engines, error) —
                        # fall through and process every collected item through
                        # the regular per-sentence engine chain, order kept.

                    for b_text, b_mid in batch:
                        if not self.is_running:
                            break
                        logger.info(f"Generating Audio for: {b_text[:30]}...")
                        output_file = self._generate_with_engine_chain(loop, b_text)

                        if self.discard_current:
                            logger.info("TTS finished, but was interrupted. Discarding audio.")
                            if output_file and os.path.exists(output_file):
                                try:
                                    os.remove(output_file)
                                except Exception:
                                    pass
                            self.queue.task_done()
                            continue

                        self._deliver_output(output_file, b_mid)
                        self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def stop(self):
        self.is_running = False
        self.discard_current = True
        self.queue.put(None)
        if hasattr(self, 'playback_worker'):
            self.playback_worker.stop()
        self.quit()
        self.wait()

class _AudioChunk:
    __slots__ = ("file_path", "text", "message_id", "persist", "is_poison")

    def __init__(self, file_path: Optional[str] = None, text: str = "",
                 message_id: Optional[str] = None, persist: bool = False,
                 is_poison: bool = False):
        self.file_path = file_path
        self.text = text
        self.message_id = message_id
        self.persist = persist
        self.is_poison = is_poison

class PipelinedTTSWorker(QThread):
    audio_ready_signal = pyqtSignal(str)
    segment_saved_signal = pyqtSignal(str, str)
    lipsync_signal = pyqtSignal(float)
    queue_empty_signal = pyqtSignal()

    AUDIO_BUFFER_SIZE = 2
    LIPSYNC_POLL_SEC = 0.05
    TTS_TIMEOUT_SEC = 30

    def __init__(self, tts_method, character_name, voice_id=None, language="en"):
        super().__init__()

        self.text_queue = queue.Queue()

        self.is_running = True
        self.discard_current = False
        self._interrupt_flag = threading.Event()

        self.tts_method = tts_method
        self.character_name = character_name
        self.voice_id = voice_id
        self.language = language

        self.configuration_settings = configuration.ConfigurationSettings()
        self.configuration_api = configuration.ConfigurationAPI()
        self.configuration_characters = configuration.ConfigurationCharacters()

        self.device_index = self.configuration_settings.get_main_setting("output_device_real_index")

        self.tts_mode = self.configuration_settings.get_main_setting("tts_voicing_mode") or 0
        self.tts_custom_regex = self.configuration_settings.get_main_setting("tts_custom_regex") or ""

        self._in_tts_quote = False
        self._in_asterisk = False

        self._tts_engines_initialized = False
        self.xtts = None
        self.edge = None
        self.kokoro = None
        self.silero = None
        self.qwen = None
        self.eleven = None
        self.comfy_tts = None
        self._fallback_warned = set()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_buffer: Optional[asyncio.Queue] = None
        self._producer_task: Optional[asyncio.Task] = None
        self._player_task: Optional[asyncio.Task] = None

    def _ensure_tts_engines(self):
        if self._tts_engines_initialized:
            return
        from app.utils.text_to_speech import (
            XTTSv2_SOW_System, EdgeTTS, KokoroTTS_SOW_System,
            SileroTTS_SOW_System, Qwen3TTS_SOW_System, ElevenLabs
        )
        self.xtts = XTTSv2_SOW_System()
        self.edge = EdgeTTS()
        self.kokoro = KokoroTTS_SOW_System()
        self.silero = SileroTTS_SOW_System()
        self.qwen = Qwen3TTS_SOW_System()
        self.eleven = ElevenLabs()
        self._tts_engines_initialized = True
        logger.info(f"[PipelinedTTS] engines initialized for method='{self.tts_method}'")

    def _get_comfy_tts_async(self):
        if self.comfy_tts is None:
            from app.utils.providers.comfy_tts import ComfyUITTS
            from app.utils.media.comfy_client import ComfyUIClient
            self.comfy_tts = ComfyUITTS(ComfyUIClient())
        return self.comfy_tts

    def add_text(self, text, message_id=None):
        if not text:
            return

        self.discard_current = False
        self._interrupt_flag.clear()

        text = self.clean_text_for_speech(text)
        if not text:
            return

        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        max_chunk_len = 450

        def _enqueue(chunk_text):
            if chunk_text:
                self.text_queue.put((chunk_text, message_id))

        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chunk_len:
                if current_chunk:
                    _enqueue(current_chunk)
                    current_chunk = ""
                sub_chunks = re.split(r'(?<=[,;])\s', sentence)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if len(sub) > max_chunk_len:
                        for i in range(0, len(sub), max_chunk_len):
                            _enqueue(sub[i:i+max_chunk_len])
                    else:
                        _enqueue(sub)
            else:
                if len(current_chunk) + len(sentence) + 1 > max_chunk_len:
                    _enqueue(current_chunk)
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

        if current_chunk:
            _enqueue(current_chunk)

    def clean_text_for_speech(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        mode = self.tts_mode
        regex = self.tts_custom_regex

        # --- Mode 0: Voice Everything ---
        if mode == 0:
            return re.sub(r"[*_~`#]", "", text).strip()

        # --- Mode 1: Voice Only Quotes ---
        elif mode == 1:
            result = []
            parts = re.split(r'(["“”«»])', text)
            for part in parts:
                if part in ('"', "“", "”", "«", "»"):
                    self._in_tts_quote = not self._in_tts_quote
                    if not self._in_tts_quote:
                        result.append(" ")
                    continue
                if self._in_tts_quote:
                    result.append(part)
            
            cleaned = re.sub(r"[*_~`#]", "", "".join(result)).strip()

            if not cleaned and not any(q in text for q in ('"', "“", "”", "«", "»")):
                no_actions = re.sub(r"\*[^*]+\*", "", text)
                no_actions = re.sub(r"_[^_]+_", "", no_actions)
                cleaned = re.sub(r"[*_~`#]", "", no_actions).strip()

            return cleaned

        # --- Mode 2: Ignore Asterisks ---
        elif mode == 2:
            result = []
            parts = re.split(r"(\*|_)", text)
            for part in parts:
                if part in ("*", "_"):
                    self._in_asterisk = not self._in_asterisk
                    if not self._in_asterisk:
                        result.append(" ")
                    continue
                if not self._in_asterisk:
                    result.append(part)
            return re.sub(r"[*_~`#]", "", "".join(result)).strip()

        # --- Mode 3: Voice Outside Quotes ---
        elif mode == 3:
            result = []
            parts = re.split(r'(["“”«»])', text)
            for part in parts:
                if part in ('"', "“", "”", "«", "»"):
                    if part in ("“", "«"):
                        self._in_tts_quote = True
                    elif part in ("”", "»"):
                        self._in_tts_quote = False
                    else:
                        self._in_tts_quote = not self._in_tts_quote

                    if not self._in_tts_quote:
                        result.append(" ")
                    continue

                if not self._in_tts_quote:
                    result.append(part)

            return re.sub(r"[*_~`#]", "", "".join(result)).strip()

        # --- Mode 4: Custom Regex ---
        elif mode == 4:
            if not regex:
                return re.sub(r"[*_~`#]", "", text).strip()
            try:
                matches = re.findall(regex, text)
                if matches:
                    extracted = []
                    for m in matches:
                        if isinstance(m, tuple):
                            val = next((g for g in m if g), "")
                            extracted.append(val)
                        else:
                            extracted.append(m)
                    return " ".join(extracted).strip()
                return ""
            except re.error:
                return re.sub(r"[*_~`#]", "", text).strip()

        return text.strip()

    def clear_queue(self):
        with self.text_queue.mutex:
            self.text_queue.queue.clear()
        self.discard_current = True
        self._interrupt_flag.set()
        self._in_tts_quote = False
        self._in_asterisk = False
        if self._loop and self._audio_buffer and self._loop.is_running():
            def _drain():
                while not self._audio_buffer.empty():
                    try:
                        self._audio_buffer.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            self._loop.call_soon_threadsafe(_drain)
        try:
            self.lipsync_signal.emit(0.0)
        except Exception:
            pass
        logger.info("[PipelinedTTS] queue cleared, playback interrupted")

    def stop(self):
        self.is_running = False
        self.discard_current = True
        self._interrupt_flag.set()
        self.text_queue.put(None)

        if self._loop and self._loop.is_running():
            for task in (self._producer_task, self._player_task):
                if task and not task.done():
                    self._loop.call_soon_threadsafe(task.cancel)

        try:
            sd.stop()
        except Exception:
            pass

        self.quit()
        self.wait()
        logger.info("[PipelinedTTS] worker stopped")

    def run(self):
        logger.info(f"PipelinedTTSWorker started (method={self.tts_method}, "
                    f"buffer={self.AUDIO_BUFFER_SIZE})")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._audio_buffer = asyncio.Queue(maxsize=self.AUDIO_BUFFER_SIZE)

        self.discard_current = False
        self._interrupt_flag.clear()

        self._producer_task = self._loop.create_task(self._producer_loop())
        self._player_task = self._loop.create_task(self._player_loop())

        try:
            self._loop.run_until_complete(
                asyncio.gather(self._producer_task, self._player_task,
                               return_exceptions=True)
            )
        except Exception as e:
            logger.error(f"[PipelinedTTS] run() error: {e}", exc_info=True)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            logger.info("[PipelinedTTS] run() exited")

    async def _producer_loop(self):
        logger.info("[PipelinedTTS] producer loop started")

        while self.is_running:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.text_queue.get(timeout=0.5)
                )
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[PipelinedTTS] producer get error: {e}")
                continue

            if item is None:

                logger.info("[PipelinedTTS] producer received poison pill")
                await self._audio_buffer.put(_AudioChunk(is_poison=True))
                break

            text, message_id = item
            if not text:
                continue

            self.discard_current = False
            self._interrupt_flag.clear()

            logger.info(f"[PipelinedTTS] producer: generating '{text[:40]}...'")

            try:
                self._ensure_tts_engines()
            except Exception as e:
                logger.error(f"[PipelinedTTS] engine init failed: {e}")
                continue

            output_file = None
            try:
                output_file = await asyncio.wait_for(
                    self._generate_wav(text),
                    timeout=self.TTS_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                logger.error(f"[PipelinedTTS] TTS timed out for: '{text[:40]}...'")
                continue
            except asyncio.CancelledError:
                logger.info("[PipelinedTTS] producer cancelled")
                break
            except Exception as e:
                logger.error(f"[PipelinedTTS] TTS generation error: {e}", exc_info=True)
                continue

            if self.discard_current:
                logger.info(f"[PipelinedTTS] producer: discarding '{text[:40]}...' (interrupted)")
                if output_file and os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception:
                        pass
                continue

            if not output_file or not os.path.exists(output_file):
                logger.warning(f"[PipelinedTTS] producer: no output file for '{text[:40]}...'")
                continue

            try:
                import base64
                b64_audio = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: base64.b64encode(open(output_file, "rb").read()).decode("utf-8")
                )
                self.audio_ready_signal.emit(b64_audio)
            except Exception as e:
                logger.error(f"[PipelinedTTS] base64 encoding error: {e}")

            if message_id:
                try:
                    relative_path = await asyncio.get_event_loop().run_in_executor(
                        None, self._persist_segment_for_replay, output_file, message_id
                    )
                    if relative_path:
                        self.segment_saved_signal.emit(message_id, relative_path)
                except Exception as e:
                    logger.error(f"[PipelinedTTS] persist error: {e}")

            chunk = _AudioChunk(
                file_path=output_file,
                text=text,
                message_id=message_id,
                persist=False,
            )
            try:
                await self._audio_buffer.put(chunk)
                logger.info(f"[PipelinedTTS] producer: enqueued '{text[:40]}...' "
                            f"(buffer now has {self._audio_buffer.qsize() + 1} items)")
            except asyncio.CancelledError:
                try:
                    if output_file and os.path.exists(output_file):
                        os.remove(output_file)
                except Exception:
                    pass
                break

        logger.info("[PipelinedTTS] producer loop ended")

    async def _player_loop(self):
        logger.info("[PipelinedTTS] player loop started")

        while self.is_running:
            try:
                chunk = await self._audio_buffer.get()
            except asyncio.CancelledError:
                logger.info("[PipelinedTTS] player cancelled")
                break
            except Exception as e:
                logger.error(f"[PipelinedTTS] player get error: {e}")
                continue

            if chunk is None or chunk.is_poison:
                logger.info("[PipelinedTTS] player received poison pill")
                break

            if not chunk.file_path or not os.path.exists(chunk.file_path):
                logger.warning(f"[PipelinedTTS] player: missing file '{chunk.file_path}'")
                continue

            if self._interrupt_flag.is_set():
                logger.info(f"[PipelinedTTS] player: skipping '{chunk.text[:40]}...' (interrupted)")
                try:
                    if not chunk.persist and os.path.exists(chunk.file_path):
                        os.remove(chunk.file_path)
                except Exception:
                    pass
                continue

            logger.info(f"[PipelinedTTS] player: playing '{chunk.text[:40]}...'")
            await self._play_audio_with_lipsync(chunk.file_path, chunk.persist)

            if self._audio_buffer.empty() and self.text_queue.empty() and not self._interrupt_flag.is_set():
                try:
                    self.queue_empty_signal.emit()
                except Exception:
                    pass

        try:
            self.lipsync_signal.emit(0.0)
        except Exception:
            pass

        logger.info("[PipelinedTTS] player loop ended")

    async def _play_audio_with_lipsync(self, file_path: str, persist: bool):
        try:
            data, samplerate = await asyncio.get_event_loop().run_in_executor(
                None, lambda: sf.read(file_path, dtype='float32')
            )
        except Exception as e:
            logger.error(f"[PipelinedTTS] read error: {e}")
            try:
                if not persist and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            return

        if data is None or len(data) == 0:
            logger.warning(f"[PipelinedTTS] empty audio data for '{file_path}'")
            try:
                if not persist and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            return

        try:
            sd.play(data, samplerate, device=self.device_index)
        except Exception as e:
            logger.error(f"[PipelinedTTS] sd.play error: {e}")
            try:
                if not persist and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
            return

        duration = len(data) / samplerate
        chunk_size = max(1, int(samplerate * self.LIPSYNC_POLL_SEC))
        total_samples = len(data)

        is_stereo = data.ndim > 1
        if is_stereo:
            data_mono = data.mean(axis=1)
        else:
            data_mono = data

        slept = 0.0
        sample_idx = 0
        try:
            while slept < duration:
                if self._interrupt_flag.is_set():
                    sd.stop()
                    self.lipsync_signal.emit(0.0)
                    logger.info(f"[PipelinedTTS] player: interrupted at {slept:.2f}/{duration:.2f}s")
                    break

                end_idx = min(sample_idx + chunk_size, total_samples)
                current_chunk = data_mono[sample_idx:end_idx]
                if len(current_chunk) > 0:
                    rms = float(np.sqrt(np.mean(current_chunk ** 2)))
                    mouth_open = min(rms * 5.0, 1.0)
                    try:
                        self.lipsync_signal.emit(mouth_open)
                    except Exception:
                        pass

                await asyncio.sleep(self.LIPSYNC_POLL_SEC)
                slept += self.LIPSYNC_POLL_SEC
                sample_idx += chunk_size

        except asyncio.CancelledError:
            sd.stop()
            try:
                self.lipsync_signal.emit(0.0)
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"[PipelinedTTS] lipsync loop error: {e}")
        finally:

            try:
                self.lipsync_signal.emit(0.0)
            except Exception:
                pass

            if not self._interrupt_flag.is_set():
                try:
                    await asyncio.get_event_loop().run_in_executor(None, sd.wait)
                except Exception:
                    pass

            try:
                if not persist and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass

    async def _generate_wav(self, text: str) -> Optional[str]:
        """Phase-B: порядок попыток задаёт media.yaml:tts_engine_order.

        Выбор GUI (self.tts_method) всегда первый; остальные элементы
        списка — фолбэки на случай ошибки/пустого результата.
        """
        from app.utils.providers.comfy_tts import arun_fallback_chain, configured_tts_chain
        return await arun_fallback_chain(
            configured_tts_chain(self.tts_method),
            lambda method: self._dispatch_engine_async(method, text),
            on_fallback=lambda cur, nxt, exc: log_tts_engine_fallback(self, cur, nxt, exc),
        )

    async def _dispatch_engine_async(self, method: str, text: str) -> Optional[str]:
        if method == "XTTSv2":
            return await self.xtts.generate_speech_with_xttsv2_sow_system(
                text, self.language, self.character_name)
        elif method == "Edge TTS":
            return await self.edge.generate_speech_with_edge_tts_sow_system(
                text, self.character_name)
        elif method == "Kokoro":
            return await self.kokoro.generate_speech_with_kokoro(
                text, self.character_name)
        elif method == "ComfyUI":
            return await self._get_comfy_tts_async().generate(text, self.character_name)
        elif method == "Silero":
            return await self.silero.generate_speech_with_silero(
                text, self.character_name)
        elif method == "Qwen-3 TTS":
            return await self.qwen.generate_speech_with_qwen3(
                text, self.character_name)
        elif method == "ElevenLabs":
            return await self.eleven.generate_speech_with_elevenlabs_sow_system(
                self.voice_id)
        else:
            logger.error(f"[PipelinedTTS] unknown TTS method: {method}")
            return None


    def _persist_segment_for_replay(self, source_file, message_id):
        if not source_file or not os.path.exists(source_file):
            return None

        segment_dir = os.path.join(
            os.getcwd(), "app", "data", ".soul",
            self.character_name, "tts_audio", message_id
        )
        os.makedirs(segment_dir, exist_ok=True)

        existing = [f for f in os.listdir(segment_dir)
                    if f.lower().endswith((".wav", ".mp3"))]
        next_index = len(existing) + 1
        ext = os.path.splitext(source_file)[1] or ".wav"
        segment_filename = f"seg_{next_index:04d}{ext}"
        destination = os.path.join(segment_dir, segment_filename)

        shutil.copyfile(source_file, destination)
        return f"tts_audio/{message_id}/{segment_filename}"
