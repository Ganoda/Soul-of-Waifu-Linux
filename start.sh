#!/usr/bin/env bash
# Soul of Waifu v2.4.7 — Linux launcher
cd "$(dirname "$0")" || exit 1

echo "============================================="
echo " Soul of Waifu v2.4.7 - Starting Application"
echo "============================================="

# [1/3] ffmpeg check (system-wide)
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: ffmpeg not found in PATH."
    echo "Install it with: sudo pacman -S ffmpeg"
    read -r -p "Press Enter to exit..."
    exit 1
fi
echo "[1/3] ffmpeg found: $(command -v ffmpeg)"

# [2/3] virtual environment
if [ ! -f ".venv/bin/python" ]; then
    echo "ERROR: .venv not found. Run installer.sh first."
    read -r -p "Press Enter to exit..."
    exit 1
fi
source .venv/bin/activate
export VIRTUAL_ENV="$PWD/.venv"

# [3/3] runtime env for Linux/Wayland specifics
export HF_HOME="$PWD/app/models/hf_cache"
# QtWebEngine needs no-sandbox without privileged helper user namespaces config:
export QTWEBENGINE_CHROMIUM_FLAGS="${QTWEBENGINE_CHROMIUM_FLAGS:---no-sandbox}"
# Prefer X11 backend via XWayland for Live2D/OpenGL stability on NVIDIA:
if [ -z "$QT_QPA_PLATFORM" ] && [ -n "$WAYLAND_DISPLAY" ]; then
    export QT_QPA_PLATFORM=xcb
fi

echo "[2/3] Virtual environment activated (.venv)"
echo "[3/3] Starting main application: main.py..."
python main.py
code=$?
if [ "$code" -ne 0 ]; then
    echo ""
    echo "CRITICAL ERROR: Application exited with code $code"
    echo "Check the 'logs/' folder for details."
    read -r -p "Press Enter to exit..."
fi
exit "$code"
