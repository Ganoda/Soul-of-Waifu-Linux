#!/usr/bin/env bash
# Detached GUI launcher for desktop entries: same env as start.sh,
# but without terminal prompts and with output routed to a log file.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

LOGDIR="logs"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/$(date +%Y-%m-%d_%H-%M-%S)_launch.log"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found in PATH." >&2
    exit 1
fi

if [ ! -f ".venv/bin/python" ]; then
    echo "ERROR: .venv not found. Run ./install.sh first." >&2
    exit 1
fi

source .venv/bin/activate
export VIRTUAL_ENV="$PWD/.venv"
export HF_HOME="$PWD/app/models/hf_cache"
# Same rationale as start.sh: NVIDIA + XWayland needs ANGLE/GL instead of the
# crash-prone Vulkan GPU path, plus GPU sandbox/compositing disabled.
export QTWEBENGINE_CHROMIUM_FLAGS="${QTWEBENGINE_CHROMIUM_FLAGS:---no-sandbox} --disable-gpu-sandbox --disable-gpu-compositing --use-angle=gl"
if [ -z "${QT_QPA_PLATFORM:-}" ] && [ -n "${WAYLAND_DISPLAY:-}" ]; then
    export QT_QPA_PLATFORM=xcb
fi

exec python main.py >>"$LOG" 2>&1
