import os
import json
import logging
import re
import time
import asyncio
import platform
import subprocess
import shutil
import socket
import shlex
from pathlib import Path

import yaml
import psutil

from app.configuration import configuration
from app.utils.ai_clients.llm_config import external_backend

logger = logging.getLogger("Local Server Manager")

# Anchored to this file so the backend directory is found independently of
# the process working directory (start.sh, IDE, systemd units, ...).
MODULE_DIR = Path(__file__).resolve().parent

# Root cause of the last failed embedded-server startup, shared with
# LocalProvider so a subsequent connection error can tell the user WHY the
# model is offline instead of a bare "Connection error".
_EMBEDDED_ERROR_TTL_SECONDS = 300
_error_recorded_at: float | None = None
_error_message: str | None = None


def record_embedded_server_error(message: str) -> None:
    """Remember why the embedded llama-server failed to come up."""
    global _error_recorded_at, _error_message
    _error_recorded_at = time.monotonic()
    _error_message = message


def clear_embedded_server_error() -> None:
    """Drop the remembered failure once the server is confirmed online."""
    global _error_recorded_at, _error_message
    _error_recorded_at = None
    _error_message = None


def recent_embedded_server_error(max_age_seconds: int = _EMBEDDED_ERROR_TTL_SECONDS) -> str | None:
    """Latest embedded-server failure message, or None if stale/absent.

    Deliberately non-consuming: several requests may hit the same outage
    and each deserves the explanation.
    """
    if _error_message is None or _error_recorded_at is None:
        return None
    if time.monotonic() - _error_recorded_at > max_age_seconds:
        return None
    return _error_message


# --- Shared UI-less manager for non-GUI call sites --------------------------
# Chat send, regenerate, NPC, translation and summary paths talk to the
# provider directly without going through soul_stage_engine's
# ensure_server_running() — so the embedded llama-server never came up and
# every request died with "Local API Connection Error". Providers bootstrap
# it through this shared instance instead; it never double-spawns because
# ensure_server_running() honours the lock file / port check the same way
# GUI-owned instances do.
_shared_manager: "LocalServerManager | None" = None


def get_shared_server_manager(ui=None) -> "LocalServerManager":
    """Process-wide LocalServerManager shared by GUI classes and providers.

    A single instance keeps ``model_loaded`` and the spawned process handle
    consistent everywhere; when a GUI class asks for the manager it (re)binds
    its UI handle, so the status indicator reflects provider-side boots too.
    """
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = LocalServerManager(ui=ui)
    elif ui is not None:
        _shared_manager.ui = ui
    return _shared_manager


async def ensure_embedded_llm_server() -> None:
    """Make sure the embedded llama-server is up before a Local LLM request.

    External backends configured in llm.yaml short-circuit: the user runs
    that server themselves, there is nothing to spawn. Startup failures are
    recorded (recent_embedded_server_error) and re-raised, so callers can
    fall through to their normal connection-error handling.
    """
    if external_backend() is not None:
        return
    await get_shared_server_manager().ensure_server_running()


# --- Multimodal projector (mmproj) discovery -------------------------------
# llama.cpp only enables vision when llama-server is given an explicit
# "--mmproj <file>". Model repos ship that projector as a second GGUF next
# to the text model, with the model name embedded in the file name:
#   Huihui-gemma-4-12B-it-abliterated-GGUF.gguf
#   Huihui-gemma-4-12B-it-abliterated-projector.gguf     (suffix style)
#   mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf            (llama.cpp prefix style)

_PROJECTOR_MARKERS = ("mmproj", "projector")
_NAME_TOKEN_SPLIT = re.compile(r"[-_. ]+")


def _name_tokens(stem: str) -> list[str]:
    """Lowercased name chunks for family matching.

    'Huihui-gemma-4-12B-it' -> ['huihui', 'gemma', '4', '12b', 'it']
    """
    return [t for t in _NAME_TOKEN_SPLIT.split(stem.lower()) if t]


def _looks_like_projector(file_name: str) -> bool:
    lowered = file_name.lower()
    return any(marker in lowered for marker in _PROJECTOR_MARKERS)


def _common_token_prefix_len(left: list[str], right: list[str]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def _family_match_score(model_tokens: list[str], candidate_tokens: list[str]) -> tuple[int, int]:
    """(token-prefix length, shared-token count) for a projector-looking file.

    Two signals, strongest first:
    - a direct token prefix ("Huihui-...-projector" next to
      "Huihui-...-GGUF");
    - shared tokens ignoring order (>= 2 to count), so
      "mmproj-<Model>-<BF16>.gguf" pairs with "<Model>-<Q4_K_M>.gguf" even
      when the projector name carries extra leading tags ("Swift") or
      trailing quant tags.

    A single shared token (usually generic: "it", "f16", "gguf") never
    matches on its own - that is what keeps a foreign family's projector
    from being grabbed when several models share one directory.
    """
    prefix = _common_token_prefix_len(model_tokens, candidate_tokens)
    shared = len(set(model_tokens) & set(candidate_tokens))
    return prefix, shared


def _merged_prefix_run(model_tokens: list[str], candidate_tokens: list[str]) -> tuple[int, int]:
    """(model tokens consumed, candidate tokens consumed) by a greedy merge.

    Handles the same family spelled with different hyphenation: the model
    "Gemma4-12B-QAT..." tokenizes as ['gemma4', '12b', ...] while its MTP
    module "mtp-gemma-4-12B-it" yields ['gemma', '4', '12b', 'it'] after
    marker stripping. Walking the candidate and concatenating consecutive
    tokens until they equal the next model token aligns 'gemma'+'4' with
    'gemma4' and '12b' with '12b' - two model tokens consumed, which plain
    prefix/shared scoring can never see. A candidate token that only
    partially covers a model token ("2b" vs "12b") stops the run, keeping
    e.g. a Gemma-4-2B draft away from a Gemma-4-12B model.
    """
    model_i = cand_i = 0
    while model_i < len(model_tokens) and cand_i < len(candidate_tokens):
        merged = ""
        attempt_start = cand_i
        while cand_i < len(candidate_tokens) and len(merged) < len(model_tokens[model_i]):
            merged += candidate_tokens[cand_i]
            cand_i += 1
        if merged != model_tokens[model_i]:
            # The final merge attempt failed; rewind so the failed tokens
            # ("2b"+"it" against "12b") are not reported as consumed.
            cand_i = attempt_start
            break
        model_i += 1
    return model_i, cand_i


def _merged_qualifies(model_tokens: list[str], candidate_tokens: list[str]) -> bool:
    """True when the merge walk matched a meaningful name part.

    Two fully consumed model tokens ("gemma4" + "12b") always count; one
    model token counts only when at least three candidate tokens merged
    into it ("gemma"+"4"+"12b" -> "gemma412b"), so a two-token partial like
    "gemma"+"2b" against "gemma42b" can never pass alone.
    """
    model_used, cand_used = _merged_prefix_run(model_tokens, candidate_tokens)
    return model_used >= 2 or (model_used >= 1 and cand_used >= 3)


def _find_family_sibling(model_path, looks_like, marker_tokens, setting: str, noun: str) -> str | None:
    """Scan the model's directory for a same-family companion GGUF.

    Shared engine behind find_mmproj_sibling (projectors) and
    find_mtp_sibling (MTP draft models): candidates are recognized by
    `looks_like`, leading marker tokens are stripped before family
    matching, and `setting`/`noun` name the llm.yaml override and the
    companion kind in the skip warning.
    """
    model = Path(model_path).expanduser()
    directory = model.parent
    if not directory.is_dir():
        return None

    model_tokens = _name_tokens(model.stem)
    best: tuple[tuple[int, int, bool], Path] | None = None
    skipped: list[str] = []

    for entry in sorted(directory.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".gguf":
            continue
        if entry.name == model.name:
            continue
        if not looks_like(entry.name):
            continue

        candidate_tokens = _name_tokens(entry.stem)
        # "<marker>-<Model>-<quant>.gguf" leads with the marker token; drop
        # it so family matching compares against the model-name part.
        while candidate_tokens and candidate_tokens[0] in marker_tokens:
            candidate_tokens = candidate_tokens[1:]

        prefix, shared = _family_match_score(model_tokens, candidate_tokens)
        merged = _merged_qualifies(model_tokens, candidate_tokens)
        if prefix <= 0 and shared < 2 and not merged:
            skipped.append(entry.name)
            continue
        score = (prefix, shared, merged)
        if best is None or score > best[0]:
            best = (score, entry)

    if best is not None:
        return str(best[1])

    if skipped:
        logger.warning(
            f"{noun}-like GGUF found next to '{model.name}' but ignored "
            f"(different model family): {', '.join(sorted(skipped))}. "
            f"Set '{setting}' in llm.yaml to force one."
        )
    return None


def find_mmproj_sibling(model_path) -> str | None:
    """Auto-detect a multimodal projector GGUF next to the model file.

    Scans the model's directory for *.gguf files that look like projectors
    ("mmproj"/"projector" in the name) and belong to the model's family by
    name. Returns the sibling path, or None to keep a text-only launch.

    A projector-looking file from a DIFFERENT family (e.g. a Qwen mmproj
    lying next to a Gemma model) is deliberately ignored - attaching it
    would break or corrupt the launch - and reported once, so the user
    learns why vision did not turn on and can set mmproj_path in llm.yaml.
    """
    return _find_family_sibling(
        model_path,
        looks_like=_looks_like_projector,
        marker_tokens=("mmproj", "mm", "proj", "projector"),
        setting="mmproj_path",
        noun="Projector",
    )


# --- MTP (multi-token prediction) draft discovery ---------------------------
# Newer models ship a small MTP module as a second GGUF; llama-server uses it
# for speculative decoding when launched with "--spec-type draft-mtp
# --model-draft <file>" (see probe_mtp_spec_type). File naming follows the
# projector convention:
#   mtp-gemma-4-12B-it.gguf                             (llama.cpp prefix style)
#   <Model>-MTP.gguf                                    (suffix style)

_MTP_MARKER_TOKENS = ("mtp", "draft")


def _looks_like_mtp(file_name: str) -> bool:
    """MTP/draft draft models are recognized by whole name tokens, not
    substrings ("mtp" inside e.g. "attempt" must not match), and projector
    files are explicitly excluded so the two scanners cannot cross-attach."""
    tokens = _name_tokens(Path(file_name).stem)
    return (
        any(token in _MTP_MARKER_TOKENS for token in tokens)
        and not _looks_like_projector(file_name)
    )


def find_mtp_sibling(model_path) -> str | None:
    """Auto-detect an MTP (multi-token prediction) draft GGUF next to the model.

    Same family rules as find_mmproj_sibling, plus merge-aware matching so
    "mtp-gemma-4-12B-it.gguf" pairs with "Gemma4-12B-QAT....gguf" despite
    the different hyphenation. A draft from a DIFFERENT family is ignored
    (and reported once) - attaching it would abort or corrupt the launch.
    """
    return _find_family_sibling(
        model_path,
        looks_like=_looks_like_mtp,
        marker_tokens=_MTP_MARKER_TOKENS,
        setting="mtp_path",
        noun="MTP",
    )


# --- MTP flag support probe --------------------------------------------------
# Unknown command-line flags abort the llama-server launch, so an MTP draft
# is only attached after the binary itself admitted to supporting MTP.
# llama.cpp has shipped two spellings: the current "draft-mtp" value and the
# earlier "mtp". The result is cached per binary path (--help costs ~100ms).

_MTP_SPEC_SUPPORT_CACHE: dict[str, str | None] = {}


def probe_mtp_spec_type(server_path) -> str | None:
    """Return the --spec-type value enabling MTP on this llama-server build.

    "draft-mtp" on current builds, "mtp" on the earlier spelling, None when
    the build does not support MTP (or the probe fails) - in which case the
    draft stays unattached instead of breaking the start-up.
    """
    path = str(server_path)
    if path in _MTP_SPEC_SUPPORT_CACHE:
        return _MTP_SPEC_SUPPORT_CACHE[path]

    spec_type = None
    try:
        probe = subprocess.run(
            [path, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        help_text = f"{probe.stdout}\n{probe.stderr}"
        if "draft-mtp" in help_text:
            spec_type = "draft-mtp"
        elif "--spec-type" in help_text and re.search(r"\bmtp\b", help_text):
            spec_type = "mtp"
    except Exception as e:
        logger.warning(f"MTP support probe failed for '{path}' ({e}); launching without MTP.")

    _MTP_SPEC_SUPPORT_CACHE[path] = spec_type
    return spec_type


class LocalServerManager:
    """
    Manages the local launch of the LLM server (llama.cpp backend).
    Handles process creation, termination, lock files, and log streaming for UI feedback.
    """

    def __init__(self, ui=None):
        self.ui = ui
        self.configuration_settings = configuration.ConfigurationSettings()
        
        self.translations = {}
        selected_language = self.configuration_settings.get_main_setting("program_language")
        match selected_language:
            case 0:
                self.load_translation("en")
            case 1:
                self.load_translation("ru")

        self.server_process = None
        self.SERVER_PORT = 48596
        self.lock_file = MODULE_DIR / "backend" / "_temp" / f"llama_server_{self.SERVER_PORT}.lock"
        
        self.restart_lock = asyncio.Lock()
        self.model_loaded = False

    def load_translation(self, language):
        file_path = f"app/translations/{language}.yaml"
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                self.translations = yaml.safe_load(file)
        else:
            self.translations = {}

    def is_port_available(self, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                result = s.connect_ex(('localhost', port))
                return result != 0
        except Exception:
            return True

    def is_server_already_running(self):
        if self.lock_file.exists():
            try:
                with open(self.lock_file, 'r') as f:
                    lock_data = json.load(f)
                    pid = lock_data.get('pid')
                if pid and psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    cmd_line = ' '.join(process.cmdline()).lower()
                    if 'llama-server' in cmd_line:
                        logger.info(f"llama-server is already running with PID {pid}")
                        return True
                    else:
                        self.lock_file.unlink(missing_ok=True)
                else:
                    self.lock_file.unlink(missing_ok=True)
            except Exception:
                self.lock_file.unlink(missing_ok=True)

        if not self.is_port_available(self.SERVER_PORT):
            logger.info(f"Port {self.SERVER_PORT} appears to be in use")
            return True
        return False

    def create_lock_file(self):
        try:
            loop_time = asyncio.get_running_loop().time()
        except RuntimeError:
            loop_time = None

        lock_data = {
            'pid': os.getpid(),
            'port': self.SERVER_PORT,
            'timestamp': str(loop_time) if loop_time is not None else 'unknown'
        }
        try:
            self.lock_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.lock_file, 'w') as f:
                json.dump(lock_data, f)
        except Exception as e:
            logger.warning(f"Failed to create lock file: {e}")

    def cleanup_lock_file(self):
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.warning(f"Couldn't delete lock file: {e}")

    def _fail_with_message(self, key, fallback):
        """Translate, record and surface a fatal startup error consistently."""
        message = self.translations.get(key, fallback)
        self.update_ui_for_server_state(False)
        record_embedded_server_error(message)
        raise RuntimeError(message)

    def resolve_server_executable(self, llm_device, llm_backend):
        """Preferred llama-server path for the selected device/backend profile.

        Path is anchored to this module so it is independent of the process
        working directory; the ".exe" suffix applies to Windows only.
        """
        exe_suffix = ".exe" if platform.system() == "Windows" else ""
        current_server = None

        match llm_device:
            case 0:
                current_server = str(MODULE_DIR / f"backend/cpu/llama-server{exe_suffix}")
            case 1:
                match llm_backend:
                    case 0:
                        current_server = str(MODULE_DIR / f"backend/vulkan/llama-server{exe_suffix}")
                    case 1:
                        current_server = str(MODULE_DIR / f"backend/cuda/llama-server{exe_suffix}")
                    case 2:
                        current_server = str(MODULE_DIR / f"backend/hip/llama-server{exe_suffix}")
                    case 3:
                        current_server = str(MODULE_DIR / f"backend/sycl/llama-server{exe_suffix}")

        return current_server

    def locate_server_binary(self, candidate):
        """Find an existing llama-server executable for the embedded launch.

        Preference order:
        1. Bundled per-profile binary (app/utils/ai_clients/backend/<profile>/);
        2. Distro-provided llama-server on PATH (llama.cpp installed
           system-wide, e.g. `pacman -S llama.cpp`).

        Device/backend settings keep selecting which flavour is preferred
        (the profile decides CPU/GPU flags), but any system binary is also
        usable: the GPU/CPU flags are applied to whatever binary is picked.
        Returns None when neither exists.
        """
        if candidate and Path(candidate).exists():
            return candidate

        exe_suffix = ".exe" if platform.system() == "Windows" else ""
        system_server = shutil.which(f"llama-server{exe_suffix}")
        if system_server:
            logger.info(
                f"Bundled server not found at '{candidate}'; "
                f"falling back to system llama-server from PATH: {system_server}"
            )
            return system_server

        return None

    def resolve_mmproj_path(self, model_path):
        """Pick a multimodal projector (mmproj) for the selected model, if any.

        Priority:
        1. Explicit `mmproj_path:` override from llm.yaml (empty by default).
           A non-existent override warns and falls through to auto-detection,
           so a typo cannot block the launch.
        2. Auto-detected sibling projector GGUF in the model's directory
           (see find_mmproj_sibling).

        Returns None - meaning a text-only launch, exactly as before mmproj
        support existed - when neither yields an existing file. Best-effort
        by design: a detection failure must never prevent the server start.
        """
        if not model_path:
            return None

        from app.utils import xconfig

        try:
            override = (xconfig.load("llm").get("mmproj_path") or "").strip()
        except Exception as e:
            logger.debug(f"Could not read llm.yaml mmproj_path ({e}); auto-detection only.")
            override = ""

        if override:
            expanded = os.path.expanduser(override)
            if os.path.exists(expanded):
                logger.info(f"Multimodal projector override from llm.yaml: {expanded}")
                return expanded
            logger.warning(
                f"mmproj_path from llm.yaml does not exist: '{override}'; "
                "falling back to auto-detection."
            )

        try:
            sibling = find_mmproj_sibling(model_path)
        except Exception as e:
            logger.warning(f"mmproj auto-detection failed ({e}); continuing text-only.")
            return None
        if sibling:
            logger.info(f"Auto-detected multimodal projector next to the model: {sibling}")
            return sibling

        return None

    def resolve_mtp_path(self, model_path):
        """Pick an MTP (multi-token prediction) draft model, if any.

        Priority mirrors resolve_mmproj_path:
        1. Explicit `mtp_path:` from llm.yaml. "off"/"none"/"false" disables
           MTP entirely - the way out when a draft misbehaves; a real path
           forces that file; empty (the default) means auto-detection. A
           non-existent path warns and falls through to auto-detection, so
           a typo cannot block the launch.
        2. Auto-detected sibling draft GGUF in the model's directory
           (see find_mtp_sibling).

        Returns None - meaning a plain non-speculative launch - when neither
        yields an existing file. Best-effort by design: a detection failure
        must never prevent the server start.
        """
        if not model_path:
            return None

        from app.utils import xconfig

        try:
            raw = xconfig.load("llm").get("mtp_path")
        except Exception as e:
            logger.debug(f"Could not read llm.yaml mtp_path ({e}); auto-detection only.")
            raw = None

        # YAML 1.1 parses a bare `off`/`no`/`false` value as boolean False;
        # honor that as an explicit disable instead of letting it fall
        # through to auto-detection as if the key were empty.
        if raw is False:
            logger.info("MTP disabled via llm.yaml (mtp_path: off).")
            return None

        override = str(raw or "").strip()

        if override:
            if override.lower() in ("off", "none", "false"):
                logger.info("MTP disabled via llm.yaml (mtp_path: off).")
                return None
            expanded = os.path.expanduser(override)
            if os.path.exists(expanded):
                logger.info(f"MTP draft override from llm.yaml: {expanded}")
                return expanded
            logger.warning(
                f"mtp_path from llm.yaml does not exist: '{override}'; "
                "falling back to auto-detection."
            )

        try:
            sibling = find_mtp_sibling(model_path)
        except Exception as e:
            logger.warning(f"MTP auto-detection failed ({e}); continuing without speculative decoding.")
            return None
        if sibling:
            logger.info(f"Auto-detected MTP draft next to the model: {sibling}")
            return sibling

        return None

    async def start_server_async(self):
        logger.info("Starting LLM server process...")
        await asyncio.sleep(1)
        
        if self.is_server_already_running():
            logger.warning("The server is already running by another process, connecting to it...")
            self.model_loaded = True
            self.update_ui_for_server_state(True)
            return

        model_path = self.configuration_settings.get_main_setting("local_llm")
        llm_device = self.configuration_settings.get_main_setting("llm_device")
        llm_backend = self.configuration_settings.get_main_setting("llm_backend")
        reasoning_mode = self.configuration_settings.get_main_setting("reasoning_mode")

        current_server = self.resolve_server_executable(llm_device, llm_backend)

        if current_server is None:
            logger.error(f"Unsupported llm_device={llm_device} / llm_backend={llm_backend} combination.")
            self._fail_with_message(
                "error_unsupported_backend",
                "Unsupported device/backend combination selected for the local model."
            )

        # Bundled per-profile binary first; distro-installed llama.cpp on
        # PATH is a valid embedded fallback (flags apply the same way).
        located_server = self.locate_server_binary(current_server)
        if located_server is None:
            logger.error(
                f"Server executable not found at: {current_server} "
                "(no system llama-server on PATH either)"
            )
            self._fail_with_message(
                "error_llama_server_not_found",
                "llama-server was not found: no bundled binary in "
                "app/utils/ai_clients/backend/ and no system-wide llama.cpp on PATH. "
                "Install llama.cpp (Arch: sudo pacman -S llama.cpp) or point llm.yaml "
                "'backend:' at an already running OpenAI-compatible server "
                "(e.g. 'backend: http://127.0.0.1:8080')."
            )
        current_server = located_server

        if not model_path or not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            self._fail_with_message(
                "error_model_not_found",
                f"Model file not found: {model_path}"
            )

        gpu_layers = self.configuration_settings.get_main_setting("gpu_layers")
        context_size = self.configuration_settings.get_main_setting("context_size")
        mlock_status = self.configuration_settings.get_main_setting("mlock_status")
        flash_attention = self.configuration_settings.get_main_setting("flash_attention_status")

        if self.server_process is not None:
            logger.warning("Server already running.")
            return

        c_arg = "0" if (context_size is None or context_size <= 0) else str(context_size)

        command = [
            current_server,
            "-m", model_path,
            "-c", c_arg,
            "--port", str(self.SERVER_PORT),
            "--parallel", "1",
        ]

        # Cross-request KV chunk reuse: the summary task and the tool-call
        # pre-pass run on the same single slot and clobber the chat cache.
        # --cache-reuse lets the next request re-attach the longest common
        # token chunks instead of re-evaluating the whole prompt.
        # Appended before custom_args, so a manual value there still wins.
        command.extend(["--cache-reuse", "256"])
        logger.info("Prompt cache chunk reuse enabled (--cache-reuse 256).")

        # Vision: attach a multimodal projector when one can be resolved for
        # this model (llm.yaml override or sibling auto-detection). Appended
        # before custom_args, so a manual "--mmproj" there still wins.
        mmproj_path = self.resolve_mmproj_path(model_path)
        if mmproj_path:
            command.extend(["--mmproj", mmproj_path])
            logger.info(f"Multimodal projector: {mmproj_path}")

        # Speculative decoding: attach an MTP draft when one can be resolved
        # for this model AND the llama-server build understands the flags -
        # unknown flags abort the launch, so the binary's own --help output
        # is probed first (probe_mtp_spec_type). Also appended before
        # custom_args, so manual spec flags there still win.
        mtp_path = self.resolve_mtp_path(model_path)
        if mtp_path:
            spec_type = probe_mtp_spec_type(current_server)
            if spec_type:
                command.extend(["--spec-type", spec_type, "--model-draft", mtp_path])
                logger.info(f"MTP draft model: {mtp_path} (--spec-type {spec_type})")
            else:
                logger.warning(
                    f"MTP draft found ({mtp_path}) but this llama-server build "
                    "does not support MTP; launching without speculative decoding."
                )

        if reasoning_mode is False:
            command.extend(["--reasoning-budget", "0", "--reasoning", "off"])
            logger.info("Reasoning mode is disabled (--reasoning off + --reasoning-budget 0). Note: some models/quants ignore this flag; a <think> tag filter downstream is still recommended as a safety net.")

        chat_template = self.configuration_settings.get_main_setting("chat_template")
        if chat_template and chat_template.strip().lower() != "auto":
            template_flag = chat_template.strip().lower()
            command.extend(["--chat-template", template_flag])
            logger.info(f"Using forced Chat Template: {template_flag}")
        else:
            command.append("--jinja")
            logger.info("Using Auto Chat Template with native Jinja engine (--jinja).")

        if llm_device == 1:
            if gpu_layers is None or gpu_layers == 0:
                gpu_layers = 20
            command.extend(["--n-gpu-layers", str(gpu_layers)])
            logger.info(f"Using GPU with {gpu_layers} layers.")

            if flash_attention:
                command.extend(["-fa", "on"])
                logger.info("Flash Attention is enabled.")
        else:
            cpu_threads = self.configuration_settings.get_main_setting("cpu_threads")
            if cpu_threads and cpu_threads > 0:
                command.extend(["--threads", str(cpu_threads), "--threads-batch", str(cpu_threads)])
                logger.info(f"Using manual CPU Threads: {cpu_threads}")

        no_mmap_status = self.configuration_settings.get_main_setting("no_mmap_status")

        if mlock_status:
            command.extend(["--load-mode", "mlock"])
            logger.info("Model will be locked in RAM (mlock enabled).")
        elif no_mmap_status:
            command.extend(["--load-mode", "none"])
            logger.info("mmap disabled (--load-mode none).")

        kv_cache_type = self.configuration_settings.get_main_setting("kv_cache_type")
        if kv_cache_type and kv_cache_type != "f16":
            command.extend(["--cache-type-k", kv_cache_type, "--cache-type-v", kv_cache_type])
            logger.info(f"Using quantized KV Cache: {kv_cache_type}")

        batch_size = self.configuration_settings.get_main_setting("llm_batch_size")
        if batch_size and batch_size > 0:
            command.extend(["--batch-size", str(batch_size), "--ubatch-size", str(batch_size)])
            logger.info(f"Using prompt batch size: {batch_size}")
        
        cpu_moe_layers = self.configuration_settings.get_main_setting("cpu_moe_layers")
        if cpu_moe_layers and cpu_moe_layers > 0:
            command.extend(["--n-cpu-moe", str(cpu_moe_layers)])
            logger.info(f"MoE Offloading: Keeping experts of {cpu_moe_layers} layers in CPU RAM.")

        custom_args_str = self.configuration_settings.get_main_setting("custom_args")
        if custom_args_str and custom_args_str.strip():
            try:
                extra_args = shlex.split(custom_args_str)
                command.extend(extra_args)
                logger.info(f"Custom arguments added to command: {extra_args}")
            except Exception as e:
                logger.error(f"Error parsing custom arguments: {e}")

        logger.info(f"Executing server command: {' '.join(command)}")

        self.create_lock_file()

        try:
            self.server_process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False
            )
        except Exception as e:
            logger.error(f"Failed to launch the server process: {e}", exc_info=True)
            self.cleanup_lock_file()
            self._fail_with_message(
                "error_server_launch_failed",
                f"Failed to launch the local server: {e}"
            )

        asyncio.create_task(self.log_stream(self.server_process.stdout, logger.info))
        asyncio.create_task(self.log_stream(self.server_process.stderr, logger.info))

        await asyncio.sleep(1)

        if self.server_process.returncode is not None and self.server_process.returncode != 0:
            logger.error("Server failed to start.")
            self.server_process = None
            self.cleanup_lock_file()
            self._fail_with_message(
                "error_server_start_failed",
                "The local server process exited immediately after launch. Check the logs for details."
            )
        else:
            logger.info("Server started successfully.")

    async def log_stream(self, stream, log_func):
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                decoded_line = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                decoded_line = str(line, encoding="utf-8", errors="ignore").strip()
            
            log_func(decoded_line)

            # Model readiness is derived from the server log itself and must
            # be tracked even without a UI (shared provider-side manager),
            # otherwise ensure_server_running() would time out on a healthy
            # server. UI progress visuals stay guarded below.
            if not self.model_loaded and (
                "listening on http" in decoded_line or "all slots are idle" in decoded_line
            ):
                self.model_loaded = True
                self.update_ui_for_server_state(True)

            if self.ui:
                try:
                    if not self.model_loaded:
                        if hasattr(self.ui, 'slide_in_status_container'):
                            self.ui.slide_in_status_container()
                        else:
                            self.ui.progressBar_llm_loading.show()
                            self.ui.loading_model_label.show()

                        if hasattr(self.ui, 'update_system_status'):
                            self.ui.update_system_status("loading")

                    if "load_model: loading model" in decoded_line and not self.model_loaded:
                        self.ui.progressBar_llm_loading.setValue(20)
                        self.ui.loading_model_label.setText(self.translations.get("model_loading_step_1", "ENGINE INIT..."))

                    elif ("load:" in decoded_line or "W load:" in decoded_line) and not self.model_loaded:
                        self.ui.progressBar_llm_loading.setValue(40)
                        self.ui.loading_model_label.setText(self.translations.get("model_loading_step_2", "VERIFYING GGUF..."))

                    elif "load_model: initializing" in decoded_line and not self.model_loaded:
                        self.ui.progressBar_llm_loading.setValue(70)
                        self.ui.loading_model_label.setText(self.translations.get("model_loading_step_4", "ALLOCATING CONTEXT..."))

                    elif "model loaded" in decoded_line and not self.model_loaded:
                        self.ui.progressBar_llm_loading.setValue(85)
                        self.ui.loading_model_label.setText(self.translations.get("model_loading_step_5", "FINALIZING..."))

                    elif ("listening on http" in decoded_line or "all slots are idle" in decoded_line) and not self.model_loaded:
                        self.ui.progressBar_llm_loading.setValue(100)
                        self.ui.loading_model_label.setText(self.translations.get("model_loading_step_6", "MODEL ONLINE"))
                    
                    if self.model_loaded and "slot launch_slot_" in decoded_line:
                        if hasattr(self.ui, 'update_system_status'):
                            self.ui.update_system_status("generating")
                    
                    if self.model_loaded and ("slot      release:" in decoded_line or "all slots are idle" in decoded_line):
                        if hasattr(self.ui, 'update_system_status'):
                            self.ui.update_system_status("online")
                    
                    if hasattr(self.ui, 'lineEdit_server'):
                        self.ui.lineEdit_server.show()
                    
                except (RuntimeError, AttributeError) as e:
                    logger.debug(f"UI element no longer exists: {e}")
                    return

    def update_ui_for_server_state(self, is_running):
        if not self.ui:
            return
        try:
            if hasattr(self.ui, 'update_system_status'):
                if is_running and self.model_loaded:
                    self.ui.update_system_status("online")
                    # Online again: a previously recorded startup failure is
                    # obsolete, do not leak it into future provider hints.
                    clear_embedded_server_error()
                else:
                    self.ui.update_system_status("offline")

            if hasattr(self.ui, 'status_container'):
                if is_running:
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(
                        3000, 
                        lambda: self.ui.slide_out_status_container() if (self.model_loaded and hasattr(self.ui, 'slide_out_status_container')) else None
                    )
                else:
                    if hasattr(self.ui, 'slide_out_status_container'):
                        self.ui.slide_out_status_container()
                    else:
                        self.ui.status_container.hide()
            else:
                if not is_running:
                    self.ui.progressBar_llm_loading.hide()
                    self.ui.loading_model_label.hide()
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"UI updating error: {e}")

    async def stop_server(self):
        if self.server_process is None:
            logger.info("Server is not running.")
            self.update_ui_for_server_state(False)
            return

        logger.info("Stopping server...")

        try:
            try:
                self.server_process.terminate()
                await self.server_process.wait()
            except ProcessLookupError:
                logger.info("Local server process was already dead (likely crashed earlier).")

            self.server_process = None
            self.model_loaded = False
            self.cleanup_lock_file()
            self.update_ui_for_server_state(False)

            logger.info("The server has been successfully stopped.")
        except Exception as e:
            logger.error(f"Error when stopping the server: {e}", exc_info=True)

    def _drain_task_result(self, task):
        """Retrieve a finished task's result defensively.

        GUI buttons fire ensure_server_running via asyncio.create_task and
        drop the handle; without draining, failures surface as
        "Task exception was never retrieved" instead of reaching anyone.
        The raise sites already log the details, so this only recovers the
        exception (marks it as handled) and puts the status indicator
        offline for good measure.
        """
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.info("Server startup task finished with an error (details logged earlier); UI marked offline.")
            self.update_ui_for_server_state(False)

    async def ensure_server_running(self):
        # Fire-and-forget callers (e.g. app/gui buttons) would otherwise
        # lose our RuntimeError entirely; draining here covers every path
        # through ensure_server_running without touching the GUI layer.
        current_task = asyncio.current_task()
        if current_task is not None and not getattr(current_task, "_sow_result_drained", False):
            current_task._sow_result_drained = True
            current_task.add_done_callback(self._drain_task_result)

        async with self.restart_lock:
            # External backend configured in llm.yaml: the app never spawns
            # its own llama-server, the user runs that server themselves.
            if external_backend() is not None:
                logger.info("External LLM backend configured in llm.yaml, skipping embedded llama-server launch.")
                self.model_loaded = True
                self.update_ui_for_server_state(True)
                return

            if self.is_server_already_running():
                self.model_loaded = True
                self.update_ui_for_server_state(True)
                return
            
            if self.server_process is None:
                self.model_loaded = False
                await self.start_server_async()

                timeout = 360
                elapsed = 0
                
                while not self.model_loaded:
                    await asyncio.sleep(1)
                    elapsed += 1
                    
                    if self.server_process and self.server_process.returncode is not None:
                        logger.error(f"Server crashed with code {self.server_process.returncode}")
                        self.cleanup_lock_file()
                        self._fail_with_message(
                            "error_server_crashed",
                            "Local server crashed. Check the logs (out of memory?)."
                        )

                    if elapsed > timeout:
                        logger.error("Server loading timed out.")
                        await self.stop_server()
                        self._fail_with_message(
                            "error_server_timeout",
                            "Server loading timed out."
                        )

    async def shutdown_server_and_model(self):
        logger.info("Shutting down server and model...")
        await self.stop_server()
        self.update_ui_for_server_state(False)

    def _log_task_exception(self, task):
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Local server task failed: {e}", exc_info=True)
            self.update_ui_for_server_state(False)

    def on_shutdown_button_clicked(self):
        task = asyncio.create_task(self.shutdown_server_and_model())
        task.add_done_callback(self._log_task_exception)
    
    def on_start_server_button_clicked(self):
        task = asyncio.create_task(self.ensure_server_running())
        task.add_done_callback(self._log_task_exception)
