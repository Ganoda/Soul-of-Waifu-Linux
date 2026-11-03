# Linux Unix-Way Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Soul-of-Waifu-Linux around "everything is a file": XDG modular YAML configs, drop-in ComfyUI workflow directories with auto-discovery, an async ComfyUIClient media engine with built-in fallback, and a local-first (zero-cloud) provider stack including a Gemma 4 Abliterated preset.

**Architecture:** Thin new modules added alongside the 58k-line legacy app. A config-root resolver redirects all JSON state to `~/.config/soul-of-waifu/`; three new YAML modules (`llm.yaml`, `media.yaml`, `ui.yaml`) drive the new provider layer; providers (LLM/TTS/STT/ImageGen) sit behind an abstract registry; `ComfyUIClient` executes exported ComfyUI API-format workflows discovered from drop-in dirs, with metadata-file or heuristic node mapping, falling back to the existing Kokoro TTS when ComfyUI is unreachable. Existing dispatch sites get exactly one new branch each.

**Tech Stack:** Python 3.11 venv (existing `.venv`), PyQt6 + qasync, aiohttp (already dep), PyYAML (already dep), faster-whisper, pytest (new dev dep).

**Spec:** User message of 2026-11-03 ("Unix Way restructure", sections 1–4). No separate spec doc; this plan embeds the spec verbatim in Global Constraints.

## Global Constraints

- All communication strictly local: default endpoints `http://127.0.0.1:8188` (ComfyUI), `http://127.0.0.1:8080` (llama-server-compatible LLM). Zero cloud required to boot or chat.
- Config root resolution order: `$SOW_CONFIG_DIR` → `$XDG_CONFIG_HOME/soul-of-waifu` → `~/.config/soul-of-waifu`. Legacy repo files (`app/configuration/*.json`) are copied there on first run, never deleted.
- Drop-in roots: `app/data/workflows/tts/`, `app/data/workflows/image/`, `app/data/workflows/video/`. Every `*.json` in them becomes a selectable preset with no Python-code change.
- Meta-file convention: `<name>.json` + optional `<name>.meta.yaml` with keys `text_node_id`, `output_node_id`.
- Async discipline: no blocking HTTP/IO calls inside the qasync event loop; ComfyUI work goes through the async client.
- Fallback rule: ComfyUI unavailable ⇒ fall back to built-in local TTS (Kokoro path already wired in `AudioPlaybackWorker`) + log warning; never crash.
- Windows/NT behavior must not regress: changes confined to Linux-affected branches.
- New UI-visible option strings: TTS method `"ComfyUI"`; image provider `"ComfyUI Workflow"`.

## Review Focus

1. ComfyUI dies mid-session after having been up ⇒ next TTS call falls back to Kokoro within one retry cycle, warning logged once, window never freezes. Pin: task 6 test `test_fallback_when_unreachable`.
2. Malformed `*.json` or `*.meta.yaml` in a drop-in dir ⇒ skipped with log entry at startup scan; other presets still discovered. Pin: task 4 tests `test_malformed_skipped`, `test_meta_bad_key_defaulted`.
3. Empty output nodes / workflow returns nothing (image gen with wrong output id) ⇒ client returns empty result list + error logged, caller branches safely. Pin: task 5 test `test_history_no_outputs_returns_empty`.
4. Offline boot: no internet anywhere ⇒ app reaches GUI, version-check failure tolerated (existing try/except preserved), models loaded from local weights only. Verify: task 9 smoke block with `--network none` checklist item (manual step, documented in docs).
5. User edits workflow endpoint port in `media.yaml` ⇒ change takes effect after restart without touching code (loader must read every field from YAML, no silent defaults past declared ones). Pin: task 5 test `test_base_url_from_config`.

---

### Task 0: Baseline commit of phase-1 port fixes

**Files:**
- Commit (already modified): `requirements-linux.txt`, `start.sh`, `app/gui/icons/*`, `app/gui/interface_signals.py`, `app/gui/sow_system_signals.py`, `app/utils/text_to_speech.py`

**Interfaces:**
- Produces: clean git baseline; later tasks produce reviewable diffs on top.

- [ ] **Step 1:** Kill any running instance: `pkill -f "python main.py"` (workdir repo; tolerate exit 1 when none).
- [ ] **Step 2:** `git status --short` — confirm expected file set only (no junk in `logs/`, `.venv/`: both gitignored; verify `resources.py`, placeholder SVGs included).
- [ ] **Step 3:** Add `.venv/` to `.gitignore` if missing. Commit all as: `fix(linux): phase-1 port — venv launcher, path separators, restored icon resources`.

### Task 1: XDG config root + legacy migration

**Files:**
- Create: `app/configuration/paths.py`, `tests/test_paths.py`, `pytest.ini`
- Modify: `app/configuration/configuration.py` (three `*_path` attribute literals near lines 17, 185, 229 → routed through `paths`; every other `open()` of `self.<x>_path` follows automatically)

**Interfaces:**
- Produces:
  - `paths.config_root() -> pathlib.Path`
  - `paths.repo_config_dir() -> pathlib.Path`
  - `paths.config_file(name: str) -> Path` (inside resolved root)
  - `paths.ensure_xdg_layout(migrate: bool = True) -> bool` (copies missing legacy JSONs; mkdirs `assets`-independent dirs used later: nothing yet)
- Consumes: environment variables `SOW_CONFIG_DIR`, `XDG_CONFIG_HOME`, `HOME`.

- [ ] **Step 1: Write failing tests** (`tmp_path` + monkeypatched env): default root = `$HOME/.config/soul-of-waifu`; `SOW_CONFIG_DIR` overrides; `ensure_xdg_layout` copies `settings.json`, `api.json`, `characters.json` from fixture repo-dir when absent at target and preserves target file if already present; `config_file` joins names.
- [ ] **Step 2:** Run `pytest tests/test_paths.py -q` → FAIL (module missing).
- [ ] **Step 3:** Implement `paths.py` (pure stdlib: os, pathlib, shutil).
- [ ] **Step 4:** `pytest tests/test_paths.py -q` → PASS.
- [ ] **Step 5:** Rewire `configuration.py`: replace `"app/configuration/settings.json"` with `str(paths.config_file("settings.json"))` etc.; add `from app.configuration import paths`; call `paths.ensure_xdg_layout()` lazily inside each class `__init__` (cheap idempotent check flag-guarded in paths module). Keep default JSON *content* untouched.
- [ ] **Step 6:** `pytest tests/test_paths.py -q` still PASS (no import cycle: configuration not imported by paths).
- [ ] **Step 7:** `pytest.ini`: `[pytest]` `testpaths = tests`, addopts `-q`. Dev dep: `uv pip install pytest` (do NOT add to requirements-linux.txt).
- [ ] **Step 8:** Commit: `feat(config): XDG config root with automatic legacy migration`.

### Task 2: Modular YAML layer (llm.yaml / media.yaml / ui.yaml)

**Files:**
- Create: `config-defaults/llm.yaml`, `config-defaults/media.yaml`, `config-defaults/ui.yaml`, `app/utils/xconfig.py`, `tests/test_xconfig.py`
- Modify: `start.sh` (export `SOW_CONFIG_DIR="$HOME/.config/soul-of-waifu"` unless already set — one guard line)

**Interfaces:**
- Produces:
  - `xconfig.load(module: str) -> XConfig`; `XConfig.get(dotted: str, default=None)`; `XConfig.reload()`; singleton cache keyed by module.
  - Seeded files on first access from `config-defaults/<module>.yaml` into config root; root path always via Task 1 `paths.config_root()`.
- Defaults file contents (exact sections, commented Russian headers allowed):
  ```yaml
  # llm.yaml
  backend: http://127.0.0.1:8080      # OpenAI-compatible: llama-server / vLLM / Ollama
  api_key: ""                          # "ollama" style dummy accepted
  model: gemma4-abliterated            # served model name
  ctx: 16384
  temperature: 0.85
  repeat_penalty: 1.05
  stop_tokens: ["<end_of_turn>"]
  presets:
    gemma4-abliterated:
      chat_template: "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
      eos_token: "<end_of_turn>"
      bos_token: "<start_of_turn>"
      # template vars follow llama.cpp conventions
  stt:
    whisper_model_path: "models/whisper/small-fp16"   # local weights; empty ⇒ library default download
    device: cuda
    compute_type: float16
  ```
  ```yaml
  # media.yaml
  comfyui:
    enabled: true
    base_url: http://127.0.0.1:8188
    timeout_s: 300
    poll_interval_s: 0.5
    cache_dir: app/cache/comfy
  workflows_dir: app/data/workflows
  tts_engine_order: [comfyui, kokoro]
  image_provider_order: [comfyui-workflow, a1111, cloud]
  video_enabled: false
  ```
  ```yaml
  # ui.yaml
  theme: system                # system|dark|light (pass-through, Phase-B hook)
  wayland:
    qt_platform: xcb           # xcb|wayland|"auto" — consumed in Task 7
  hotkeys:
    panic_hide: ctrl+alt+h     # reserved; not bound yet (Phase-B)
  window_geometry_save: true
  ```

- [ ] **Step 1: Failing tests**: seeding creates file at config root; second `load()` does NOT overwrite user-edited value; `get("comfyui.base_url")` returns default; unknown dotted path returns supplied default; `reload()` picks up external edit; YAML syntax error in user file ⇒ log warning + return defaults (no exception).
- [ ] **Step 2:** Run `pytest tests/test_xconfig.py -q` → FAIL.
- [ ] **Step 3:** Implement `xconfig.py` (PyYAML safe_load; attrs read-only snapshot; exceptions swallowed→warn via stdlib logging).
- [ ] **Step 4:** PASS.
- [ ] **Step 5:** Write the three `config-defaults/*.yaml` files byte-exact as above.
- [ ] **Step 6:** PASS again (seeding exercised against real files with tmp config root).
- [ ] **Step 7:** `start.sh` guard line + commit: `feat(config): modular llm/media/ui YAML with drop-in defaults`.

### Task 3: Provider registry (Base* abstractions)

**Files:**
- Create: `app/utils/providers/__init__.py`, `app/utils/providers/base.py`, `app/utils/providers/registry.py`, `tests/test_registry.py`

**Interfaces:**
- Produces (all async-first where IO involved):
  ```python
  class BaseLLM:       async def chat(self, messages: list[dict], *, stream=False, **params) -> str | AsyncIterator[str]
  class BaseTTS:       async def generate(self, text: str, character_name: str) -> str | None   # abs path to wav/file or None
  class BaseSTT:       def transcribe(self, wav_path: str | Path, language: str | None = None) -> str   # blocking-internal; called off-loop by callers
  class BaseImageGen:  async def generate(self, prompt: str, negative: str = "", **kw) -> bytes | None
  # kind constants: KIND_LLM/KIND_TTS/KIND_STT/KIND_IMAGE == "llm"/"tts"/"stt"/"image"
  registry.register(kind: str, name: str, factory: Callable[..., object]) -> None     # stores factories, NOT instances
  registry.create(kind: str, name: str, **kwargs) -> object
  registry.available(kind: str) -> list[str]
  registry.get_factory(kind, name)
  ```
- Consumes: nothing runtime-critical (registry is standalone) — task ordering is convenience only.

- [ ] **Step 1: Failing tests**: register/create roundtrip passes kwargs; duplicate `(kind,name)` overwrite replaces silently; `available` sorted; unknown name raises `KeyError`; ABC instantiation directly raises `TypeError` for a subclass omitting an abstract method.
- [ ] **Step 2:** FAIL → implement (`abc.ABC` module; registry as dict-dict) → PASS.
- [ ] **Step 3:** Commit: `feat(providers): abstract provider registry (BaseLLM/BaseTTS/BaseSTT/BaseImageGen)`.

### Task 4: Workflow drop-in dirs + loader/discovery with meta mapping

**Files:**
- Create: `app/data/workflows/tts/README.md`, `app/data/workflows/image/README.md`, `app/data/workflows/video/README.md`, `app/data/workflows/tts/example.json`, `app/data/workflows/tts/example.meta.yaml`, `app/utils/media/__init__.py`, `app/utils/media/workflow_loader.py`, `tests/test_workflow_loader.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class WorkflowSpec:
      id: str            # filename stem
      path: Path         # json path
      category: str      # "tts" | "image" | "video"
      text_node_id: str | None
      text_input_field: str        # detected input key, e.g. "text"/"value"
      output_node_id: str | None
      source_meta: bool            # True ⇒ ids came from meta.yaml
      raw: dict                    # parsed API-format json
  workflow_loader.scan(category: str, *, base_dir: str | Path | None = None) -> list[WorkflowSpec]
  ```
- Discovery rules: `scan(category)` lists `*.json` under `workflows/<category>/` (dir from `media.yaml workflows_dir`), skipping files failing `json.loads` (logged INFO, others continue). Meta file `<stem>.meta.yaml` wins over heuristics. Heuristic (only algorithm body prescribed here):
  1. text node = first dict-order entry whose `class_type` contains substring in order `PrimitiveString` → `CLIPTextEncode` → case-insensitive contains `textto`/`tts`; its text input key = first key among `inputs` matching ∈ {"text","string","value","prompt"}.
  2. output node = first entry whose `class_type` matches regex `Save(Audio|Image)|VHS_VideoCombine$` else first node whose `_meta.title` matches same; id keys are top-level string node IDs as stored by ComfyUI API format (`{"3": {"class_type": ..., "inputs": {...}}}`).
- Example payload files: minimal valid API-format stubs, 2-node graph `"3"` CLIPTextEncode-ish + `"9"` SaveAudio-ish (`example.meta.yaml` pins `text_node_id: "3"`, `output_node_id: "9"`) so offline tests/demos run without real ComfyUI exports.
- README.md (each dir): how to save workflow in "API format" from ComfyUI ("Save (API Format)" button), naming rules, meta-file reference table, that files appear app-side instantly after restart of the scan (no restart of whole app needed only in Phase-B; today scan happens per-call/task 6 — decision: **live scan per call**, cost negligible ≤ dozens of files).

- [ ] **Step 1: Failing tests** (fixtures under `tests/fixtures/workflows/…`): happy path discovers spec w/ meta ids; heuristic-only case detects both nodes correctly on crafted graph; malformed json skipped & sibling still found; extra unknown meta key ignored; missing meta entirely ⇒ detected=false-none; empty dir ⇒ [].
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Drop-in dirs + README + examples committed: `feat(media): drop-in workflow dirs with meta-driven discovery`.

### Task 5: Async ComfyUIClient

**Files:**
- Create: `app/utils/media/comfy_client.py`, `tests/test_comfy_client.py`

**Interfaces:**
- Consumes: Task 2 `xconfig.load("media").get("comfyui.*")`, Task 4 `WorkflowSpec`/dict-or-spec.
- Produces:
  ```python
  class ComfyUIUnavailable(Exception)
  class ComfyUIClient:
      __init__(self, base_url: str | None = None, timeout_s: int | None = None,
               poll_interval_s: float | None = None, cache_dir: str | None = None)
          # None args ⇒ xconfig media defaults
      @property healthy(self) -> bool                     # last-attempt state, sticky between refreshes
      async def ping(self) -> bool                        # GET /system_stats ≤2s; sets unhealthy flag on fail
      async def run(self, workflow: Mapping | WorkflowSpec, *,
                    node_inputs: Mapping[str, Mapping[str, Any]] = {},   # deep-merge patch applied to workflow copy
                    prefix: str = "job") -> dict          # {"audio": [Path], "images": [Path], "video": [Path]}
  ```
  Execution loop: validate ComfyUI reachable via `ping` unless trusted-sticky → `POST /prompt` `{prompt: patched_workflow, client_id}` → poll `GET /history/{prompt_id}` until entry appears with outputs or total timeout (both configurable) → for each artifact in outputs collect `(filename, subfolder, type)` triples → `GET /view?filename=&subfolder=&type=` streamed to `cache_dir/<prefix>_<ts>_<n>.<ext>` (ext sniffed from mime `content-type` header, fallback from filename) → return grouped abs Paths. One shared `aiohttp.ClientSession` created inside running loop (lazy, reused; loop-assumption errors caught → recreate). All failure modes surface as `ComfyUIUnavailable` (connect/timeouts/HTTP≥500) except "executed with no outputs" which returns empty dict — distinguishing input for Review-Focus-3 callers.
- Cache note: never write outside repo root by default; `cache_dir` param lets tests pass `tmp_path`.

- [ ] **Step 1: Failing tests** against `aiohttp.test_utils.TestServer` stub implementing `/system_stats`, `/prompt`, `/history/{id}` (first poll → `{}`, second → done-with-audio+image entries), `/view` returning fixed bytes; asserts: run returns 1 audio + 1 image path, file contents equal served bytes, patched input value visible in captured `/prompt` body, history-no-outputs→empty dict, server-down→`ComfyUIUnavailable`, `ping` False fast, config plumbing (`test_base_url_from_config` constructs client via xconfig tmp root with custom port).
- [ ] **Step 2:** FAIL → implement → PASS. (pytest + asyncio.run wrappers; no pytest-asyncio dep.)
- [ ] **Step 3:** Commit: `feat(media): async ComfyUI client (prompt/poll/view with local caching)`.

### Task 6: Concrete providers + fallback policy

**Files:**
- Create: `app/utils/providers/openai_compat_llm.py`, `app/utils/providers/kokoro_adapter.py`, `app/utils/providers/comfy_tts.py`, `app/utils/providers/whisper_stt.py`, `app/utils/providers/comfy_image.py`, `tests/test_providers.py`

**Interfaces:**
- Consumes: registry (T3), client (T5), loader (T4), xconfig (T2); existing legacy classes imported lazily: `app.utils.text_to_speech.KokoroHandlerName` — actual symbol TBD during implementation by grepping class owning `generate_speech_with_kokoro` (grep `"class .*:" app/utils/text_to_speech.py`); adapter wraps a module-level accessor so the heavy model loads only on first generate.
- Produces:
  ```python
  OpenAICompatLLM(base_url, api_key="", model="")                       # POST {base_url}/chat/completions non-stream first
  KokoroAdapter(tts_singleton_getter)                                   # delegates to legacy method chain producing wav path
  ComfyUITTS(client, workflow_spec|None)                                # generate(): live-scan tts dir → first spec;
                                                                        # missing/none → RuntimeError("no tts workflow")
  FasterWhisperSTT(model_path=None, device=None, compute_type=None)     # model_path None/"" ⇒ media.yaml stt.* ; local_files_only=True when path set
  ComfyUIImageGen(client, workflow_spec|None)                           # maps prompt→text node, returns first image bytes
  register_defaults()                                                   # registers all five names below
  registry.create("tts", "comfyui"), ("tts","kokoro"), ("llm","openai-compat"), ("stt","faster-whisper"), ("image","comfyui")
  ```
  Fallback policy encoded as function (used by UI wiring in Task 7):
  ```python
  pick_tts_engine(xcfg_media) -> ["comfyui"|"kokoro"] ordered list trimmed to registered+healthy  # engine_order semantics
  ```

- [ ] **Step 1: Failing tests** (fake server reused): LLM happy-path request shape (Authorization header only when api_key non-empty; model echoed back from response payload); STT constructed-but-not-loaded on init with config-provided path forwarded (mock `WhisperModel` via injected `model_cls` kwarg — constructor accepts hidden `model_cls=faster_whisper.WhisperModel`); ComfyUITTS returns file path end-to-end against stub `/view`; unreachable ⇒ `ComfyUIUnavailable` propagates; `pick_tts_engine` trims dead engines preserving order.
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Commit: `feat(providers): concrete local-first providers with ComfyUI fallback policy`.

### Task 7: Wire dispatch sites (minimal-touch)

**Files:**
- Modify: `app/utils/text_to_speech.py` — two dispatcher zones: `AudioPlaybackWorker.run` branch chain (~lines 1105–1132) and second method-string switch (~line 1719 area, exact locator: `elif method == "Edge TTS"`); lazy `self.comfy_tts = None` init in the worker ctor + single `_get_comfy()` builder
- Modify: `app/utils/image_generator.py` — constant `PROVIDER_COMFY_WORKFLOW = "ComfyUI Workflow"` + one `elif` in `generate_image` before cloud fallbacks
- Modify: `app/utils/speech_to_text.py` — `SpeechToText.__init__` (~line 155) forwards `model_size/device/compute_type` from `xconfig.get("llm.stt.…")` when keys non-empty
- Modify: `main.py` — pre-QApplication block: read `ui.yaml`; map `wayland.qt_platform: "auto"` ⇒ unset `QT_QPA_PLATFORM`, else set env var; also `paths.ensure_xdg_layout()` invoked once before any Qt window writes settings

**Interfaces:**
- Consumes: T5/T6 providers; `pick_tts_engine` ordering.
- Produces: user-visible options `"ComfyUI"` (TTS dropdown value), `"ComfyUI Workflow"` (image provider); behavioral contract per Review Focus items 1–3:
  - ComfyUI branch: try provider; on `ComfyUIUnavailable` or empty result → log warning `logger.warning("[Media] ComfyUI unavailable, falling back to Kokoro")` → immediately invoke the existing `self.kokoro.generate_speech_with_kokoro(...)` branch inline (no requeue); sticky-unhealthy suppression avoids warn-spam (per-process flag).
  - On success the produced file flows through unchanged discard/cleanup logic below the dispatch site.
- Do-NOT: touch anything else in the two giant GUI files besides exact insert points listed above.

- [ ] **Step 1: Failing test?** — dispatch lives inside Qt-heavy code; regression coverage via import-level test: `tests/test_wiring_import.py` asserting the two new string literals exist in the switched sources (`"ComfyUI"` in worker chain; class branches constructible headless). Run app afterwards for manual verification instead of full-GUI automation.
- [ ] **Step 2:** Implement inserts exactly.
- [ ] **Step 3:** Full `pytest -q` (all tasks green) then GUI smoke: launch via `./start.sh`, confirm window opens, logs contain `[Config] root=...soul-of-waifu` INFO line and NO ERROR lines from new modules; close gracefully.
- [ ] **Step 4:** Commit: `feat(integration): wire ComfyUI TTS/image + config-driven STT/platform into dispatch sites`.

### Task 8: Gemma 4 Abliterated preset + docs

**Files:**
- Create: `docs/gemma-preset.md`, `docs/linux-install.md` (fold phase-1 knowledge: uv/python3.11, CUDA torch line, playwright chromium, fairseq caveat resolved by rvc-optional import)
- Modify: `config-defaults/llm.yaml` — ensure `presets.gemma4-abliterated` block present + `# Set model path after GGUF placement:` comment pointing at docs

**Interfaces:**
- Produces: operator-ready instructions; llm.yaml documents the exact turn-token template (values already fixed in Task 2 template):
  `<start_of_turn>user\n … <end_of_turn>\n<start_of_turn>model\n`, EOS `<end_of_turn>`.
- Steps (doc content, not code):
  1. Obtain local GGUF (example search tag `gemma-*-abliterated GGUF Q4_K_M`), place under `~/Soul-of-Waifu-Linux/assets/local_llm/` or reuse existing llama-server instance.
  2. Launch llama-server with `--host 127.0.0.1 --port 8080 --ctx-size 16384` (or set `backend:` accordingly for vLLM/Ollama variants).
  3. In-app path: Options → LLM Settings → choose `Local (llama.cpp/OpenAI-compat)`; backend URL stays `http://127.0.0.1:8080`.
  4. Verify zero-network: run with internet disabled; model loads from disk; version-check banner tolerated.

- [ ] **Step 1:** Write both docs (plain markdown, table of env knobs mirroring yaml files).
- [ ] **Step 2:** Sanity-render check: `grep -c "start_of_turn" docs/gemma-preset.md` ≥ 3.
- [ ] **Step 3:** Commit: `docs: Gemma 4 Abliterated preset guide + consolidated linux install notes`.

### Task 9: Final verification sweep

**Files:** none modified.

- [ ] **Step 1:** `pytest -q` from repo root → all green count recorded.
- [ ] **Step 2:** Kill leftovers; fresh `./start.sh` ≥60 s uptime; confirm in newest log: config root line, no traceback, TTS worker started.
- [ ] **Step 3:** Offline spot-check: `env -i HOME=$HOME PATH=/usr/bin WAYLAND_DISPLAY=$WAYLAND_DISPLAY DISPLAY=:0 ./start.sh` limited-AUTH variant too aggressive ⇒ do soft-offline instead: temporarily unset proxy vars only; document manual full-offline test pointer for user.
- [ ] **Step 4:** (Manual) user confirms window renders & responds; user feedback appended to commit trailer not required.
- [ ] **Step 5:** Tag the milestone: `git tag linux-way-m1` (annotated) once user approves smoke.

## Self-Review Notes

- Spec §1 configs + XDG → Tasks 1–2. Spec §1 drop-ins → Task 4 (+client consumption 5–6). Spec §2 local LLM/Gemma → Tasks 2 (template), 6 (provider), 8 (preset/docs). Spec §2 STT local weights → Tasks 2 (`stt:`), 6 (`FasterWhisperSTT`), 7 (wire). Spec §3 client/mapping/full-cycle/fallback → Tasks 4–7. Spec §4 registry → Task 3. Runtime verification → Task 9. Covered.
- Types checked across tasks: `WorkflowSpec` frozen dataclass identical in T4/T5/T6 usages; `run_workflow` renamed consistently to `client.run(...)` everywhere; `pick_tts_engine` defined once (T6), referenced T7.
- Proportion kept: bodies specified only for heuristics/discovery order (non-obvious) and polling contract (error taxonomy), everything else signature-level.
