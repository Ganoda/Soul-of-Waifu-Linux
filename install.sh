#!/usr/bin/env bash
# =============================================================================
# Soul of Waifu v2.4.7 — Linux/Unix installer
#
# Creates .venv, installs pinned dependencies from requirements-linux.txt
# (including the documented deferred --no-deps packages and the one-shot
# fairseq patch), then makes sure Playwright Chromium and system ffmpeg
# are available. Idempotent: safe to re-run at any time; already-installed
# pins are skipped by pip/uv, the fairseq patch only applies once.
#
# Full manual procedure and troubleshooting: docs/linux-install.md
#
# Usage:
#   ./install.sh              # interactive: CUDA (default) / CPU
#   ./install.sh --cpu        # force CPU-only PyTorch build
#   ./install.sh --cuda       # force CUDA (cu128) PyTorch build
#   ./install.sh --recreate   # wipe and recreate .venv
#   ./install.sh --yes        # assume defaults everywhere (non-interactive)
# Environment overrides: SOW_GPU=cuda|cpu, SOW_ASSUME_YES=1
# =============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$APP_DIR"

TORCH_VERSION="2.10.0"
REQUIREMENTS_FILE="requirements-linux.txt"
RVC_PYTHON_PIN="rvc-python @ git+https://github.com/JarodMica/rvc-python@9a67ac77743d749e9f2cbbb02798957d5343fd19"
SPACY_MODEL_URL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"

GPU_MODE="${SOW_GPU:-}"
ASSUME_YES="${SOW_ASSUME_YES:-0}"
RECREATE=0
QUIET_PIP="--quiet"
NO_CACHE=(--no-cache-dir)

log()  { echo "[installer] $*"; }
warn() { echo "[installer] WARNING: $*" >&2; }

usage() {
    sed -n '2,26p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "[installer] ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --cpu)    GPU_MODE="cpu" ;;
        --cuda)   GPU_MODE="cuda" ;;
        --yes|-y) ASSUME_YES=1 ;;
        --recreate|-r) RECREATE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "Unknown option: $1 (try --help)" ;;
    esac
    shift
done

ask_default() { # ask_default <y|n> <prompt>  -> sets REPLY ("" = non-interactive)
    REPLY=""
    if [ "$ASSUME_YES" = "1" ] || [ ! -t 0 ]; then
        REPLY=""
        return 0
    fi
    read -r -p "$2" REPLY
    [ -z "$REPLY" ] && REPLY="$1"
    return 0
}

# ---------------------------------------------------------------------------
echo "=============================================================="
echo " Soul of Waifu v2.4.7 — Linux installer"
echo " Repo: ${APP_DIR}"
echo " Guide: docs/linux-install.md"
echo "=============================================================="

# ---------------------------------------------------------------------------
echo
log "[1/8] Checking system prerequisites..."

PYTHON_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    log "  uv found: ${UV_BIN}"
else
    UV_BIN=""
    log "  uv not found — falling back to python -m venv + pip"
fi

for cand in python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"
        break
    fi
done
[ -n "$PYTHON_BIN" ] || die "Neither python3.11 nor python3 found in PATH. Install Python 3.11 first."

PY_FULL="$(${PYTHON_BIN} -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
PY_MAJOR="${PY_FULL%%.*}"
PY_REST="${PY_FULL#*.}"
PY_MINOR="${PY_REST%%.*}"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    die "Python ${PY_FULL} is too old. Soul of Waifu requires Python >= 3.10 (3.11 recommended)."
fi
if [ "$PY_MAJOR.$PY_MINOR" != "3.11" ]; then
    warn "Python ${PY_FULL} found. 3.11 is recommended; upstream pins were validated on 3.11."
fi
log "  python: ${PYTHON_BIN} (${PY_FULL})"

if command -v git >/dev/null 2>&1; then
    log "  git found: $(command -v git)"
else
    GIT_MISSING=1
    warn "  git not found — needed to pull the pinned rvc-python dependency from GitHub."
fi

# ffmpeg is required at runtime by start.sh; try a helpful guided install.
if command -v ffmpeg >/dev/null 2>&1; then
    log "  ffmpeg found: $(command -v ffmpeg)"
else
    warn "ffmpeg not found in PATH (required for voice/video features)."
    PM=""
    if command -v pacman >/dev/null 2>&1; then PM="pacman"
    elif command -v dnf >/dev/null 2>&1; then PM="dnf"
    elif command -v apt-get >/dev/null 2>&1; then PM="apt-get"
    elif command -v zypper >/dev/null 2>&1; then PM="zypper"
    fi
    if [ -n "$PM" ]; then
        INSTALL_CMD=""
        case "$PM" in
            pacman)  INSTALL_CMD="sudo pacman -S --noconfirm ffmpeg" ;;
            dnf)     INSTALL_CMD="sudo dnf install -y ffmpeg" ;;
            apt-get) INSTALL_CMD="sudo apt-get install -y ffmpeg" ;;
            zypper)  INSTALL_CMD="sudo zypper --non-interactive install ffmpeg" ;;
        esac
        log "  Suggested command: ${INSTALL_CMD}"
        ask_default "n" "  Attempt automatic install now? [y/N] "
        if [ "$REPLY" = "y" ] || [ "$REPLY" = "Y" ]; then
            if sh -c "$INSTALL_CMD"; then
                :
            else
                warn "Auto-install failed; install ffmpeg manually and re-run ./install.sh later."
            fi
        fi
    else
        log "  Install ffmpeg via your package manager, then re-run ./install.sh if you need voice/video."
    fi
    command -v ffmpeg >/dev/null 2>&1 || warn "Continuing WITHOUT ffmpeg — voice/video features will not work."
fi

# ---------------------------------------------------------------------------
echo
log "[2/8] Preparing Python virtual environment (.venv)..."

CREATED=0
if [ -d ".venv" ] && [ "$RECREATE" = "1" ]; then
    log "  Removing existing .venv (--recreate)"
    rm -rf .venv
fi
if [ ! -f ".venv/bin/python" ]; then
    CREATED=1
    if [ -n "$UV_BIN" ]; then
        log "  Creating .venv with uv (Python 3.11, downloads a managed interpreter if needed)..."
        "$UV_BIN" venv --python 3.11 .venv \
            || "$UV_BIN" venv .venv
    else
        TARGET_PY="$PYTHON_BIN"
        if command -v python3.11 >/dev/null 2>&1; then
            TARGET_PY="python3.11"
        fi
        log "  Creating .venv with ${TARGET_PY} -m venv ..."
        ${TARGET_PY} -m venv .venv
    fi
else
    log "  Reusing existing .venv (delete it or pass --recreate for a clean rebuild)."
fi

VENV_PY="$APP_DIR/.venv/bin/python"
[ -x "$VENV_PY" ] || die ".venv/bin/python is missing after environment creation."

VENV_PY_SHORT=".venv/bin/python"
log "  Environment ready: ${APP_DIR}/.venv"

pip_install() {
    if [ -n "$UV_BIN" ]; then
        "$UV_BIN" pip install --python "$VENV_PY" "$QUIET_PIP" "$@"
    else
        "$VENV_PY" -m pip install "$QUIET_PIP" "${NO_CACHE[@]}" "$@"
    fi
}

# ---------------------------------------------------------------------------
echo
log "[3/8] Bootstrapping packaging tools..."
if [ -n "$UV_BIN" ]; then
    log "  Skipping pip upgrade (uv manages its own resolver)."
else
    "$VENV_PY" -m pip install --upgrade --no-cache-dir pip setuptools wheel \
        || warn "Could not upgrade pip/setuptools/wheel — continuing with bundled versions."
fi

# ---------------------------------------------------------------------------
echo
log "[4/8] Selecting and installing PyTorch ${TORCH_VERSION}..."

install_torch_variant() {
    if [ -z "$GPU_MODE" ]; then
        if [ -t 0 ]; then
            echo "    [1] CUDA (cu128) — NVIDIA GPU, 6+ GB VRAM (default)"
            echo "    [2] CPU only     — any hardware, slower"
            read -r -p "    Choose PyTorch build (1/2, default 1): " TORCH_CHOICE
            [ -z "$TORCH_CHOICE" ] && TORCH_CHOICE="1"
            case "$TORCH_CHOICE" in
                1) GPU_MODE="cuda" ;;
                2) GPU_MODE="cpu" ;;
                *) GPU_MODE="cuda" ;;
            esac
        else
            GPU_MODE="cpu"
            log "  Non-interactive mode without SOW_GPU — defaulting to the CPU build (pass SOW_GPU=cuda or --cuda for the CUDA build)."
        fi
    fi

    case "$GPU_MODE" in
        cuda)
            log "  Installing CUDA (cu128) build..."
            if [ -n "$UV_BIN" ]; then
                "$UV_BIN" pip install --python "$VENV_PY" "$QUIET_PIP" \
                    "torch==${TORCH_VERSION}" "torchvision==0.25.0" "torchaudio==${TORCH_VERSION}" \
                    --extra-index-url https://download.pytorch.org/whl/cu128
            else
                "$VENV_PY" -m pip install "$QUIET_PIP" "${NO_CACHE[@]}" \
                    "torch==${TORCH_VERSION}" "torchvision==0.25.0" "torchaudio==${TORCH_VERSION}" \
                    --extra-index-url https://download.pytorch.org/whl/cu128
            fi
            ;;
        cpu)
            log "  Installing CPU build from PyPI..."
            pip_install "torch==${TORCH_VERSION}" "torchvision==0.25.0" "torchaudio==${TORCH_VERSION}"
            ;;
        *)
            die "Unsupported SOW_GPU value '${GPU_MODE}' (expected cuda|cpu)."
            ;;
    esac
}

EXISTING_TORCH="$("$VENV_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)"
if [ "$EXISTING_TORCH" = "$TORCH_VERSION" ]; then
    log "  torch ${TORCH_VERSION} already present in .venv — skipping PyTorch install."
else
    [ -n "$EXISTING_TORCH" ] && log "  Found incompatible torch ${EXISTING_TORCH}; replacing with ${TORCH_VERSION}."
    install_torch_variant
fi

# ---------------------------------------------------------------------------
echo
log "[5/8] Installing pinned dependencies from ${REQUIREMENTS_FILE}..."
pip_install -r "$REQUIREMENTS_FILE"

# ---------------------------------------------------------------------------
echo
log "[6/8] Installing deferred packages (--no-deps, see docs/linux-install.md)..."

if pip_install --no-deps fairseq==0.12.2 pyautogui; then
    log "  fairseq + pyautogui: OK"
else
    warn "fairseq/pyautogui installation reported an error — re-run ./install.sh to retry."
fi

if [ "${GIT_MISSING:-0}" = "1" ]; then
    warn "Skipping rvc-python (git not found). Install git and re-run ./install.sh."
else
    if pip_install --no-deps "$RVC_PYTHON_PIN"; then
        log "  rvc-python (pinned commit): OK"
    else
        warn "rvc-python installation reported an error — RVC features may be unavailable."
    fi
fi

if pip_install --no-deps "$SPACY_MODEL_URL"; then
    log "  spacy model en_core_web_sm-3.7.1: OK"
else
    warn "spacy model installation reported an error — re-run ./install.sh to retry."
fi

# ---------------------------------------------------------------------------
echo
log "[7/8] Applying fairseq dataclass patch (once) and ensuring Playwright Chromium..."

SITE_PACKAGES="$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
FAIRSEQ_CONFIG="${SITE_PACKAGES}/fairseq/dataclass/configs.py"
if [ ! -f "$FAIRSEQ_CONFIG" ]; then
    warn "fairseq not found at ${FAIRSEQ_CONFIG} — skipping patch (check step [6/8])."
elif grep -Eq '^[[:space:]]+[[:alnum:]_]+: [[:alnum:]_]+Config = [[:alnum:]_]+Config\(\)[[:space:]]*$' "$FAIRSEQ_CONFIG"; then
    "$VENV_PY" - "$FAIRSEQ_CONFIG" <<'PYEOF'
import re
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()
pat = re.compile(
    r"^(\s+)(\w+):\s*(\w+Config)\s*=\s*\3\(\)\s*$",
    re.M,
)
patched = pat.sub(
    lambda m: "{}{}: {} = field(default_factory={})".format(
        m.group(1), m.group(2), m.group(3), m.group(3)
    ),
    src,
)
open(path, "w", encoding="utf-8").write(patched)
print("[installer]  fairseq patched:", path)
PYEOF
else
    log "  fairseq already patched (or upstream fixed it) — nothing to do."
fi

log "  Ensuring Playwright Chromium browser (skips if already downloaded)..."
if "$VENV_PY" -m playwright install chromium; then
    log "  Playwright Chromium: OK"
else
    warn "'playwright install chromium' failed (offline?) — the browser agent needs it; retry later with:"
    warn "    ${VENV_PY_SHORT} -m playwright install chromium"
fi

# ---------------------------------------------------------------------------
echo
log "[8/8] Final checks..."
PIP_CHECK_BIN=""
if [ -n "$UV_BIN" ] && command -v uv >/dev/null 2>&1; then
    uv pip check --python "$VENV_PY" && log "  dependency check: OK" \
        || warn "dependency check reported conflicts (RVC/Coqui ecosystems may conflict); the app may still run fine."
else
    "$VENV_PY" -m pip check && log "  dependency check: OK" \
        || warn "pip check reported conflicts (RVC/Coqui ecosystems may conflict); the app may still run fine."
fi

if "$VENV_PY" -c "import torch, numpy, PyQt6; print('[installer]  core imports OK')"; then
    :
else
    warn "Core import check failed — inspect the pip output above."
fi

echo
echo "=============================================================="
echo "  Installation finished!"
echo "=============================================================="
echo
echo "Next steps:"
echo "  1. Start the app:                ./start.sh"
echo "  2. Configs (Unix way) live in:   ~/.config/soul-of-waifu/"
echo "     Defaults are seeded on first run from ./config-defaults/"
echo
echo "Troubleshooting & manual procedure: docs/linux-install.md"

if [ "$ASSUME_YES" = "1" ]; then
    exit 0
fi
ask_default "n" "Start the application now? [y/N] "
if [ "$REPLY" = "y" ] || [ "$REPLY" = "Y" ]; then
    exec ./start.sh
else
    log "Done."
fi
