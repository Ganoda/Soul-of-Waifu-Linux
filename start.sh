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
    echo "ERROR: .venv not found. Run ./install.sh first."
    read -r -p "Press Enter to exit..."
    exit 1
fi
source .venv/bin/activate
export VIRTUAL_ENV="$PWD/.venv"

# [3/3] runtime env for Linux/Wayland specifics
# XDG-корень резолвится в app/configuration/paths.py
# (SOW_CONFIG_DIR > XDG_CONFIG_HOME > ~/.config/soul-of-waifu);
# здесь ничего не форсируем, чтобы не перекрывать XDG_CONFIG_HOME.
export HF_HOME="$PWD/app/models/hf_cache"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# QtWebEngine needs no-sandbox without privileged helper user namespaces config.
# NVIDIA + XWayland: Chromium reports "GBM is not supported ... Fallback to Vulkan
# rendering" and the ANGLE-Vulkan GPU process crashes, taking the whole app down
# when the VRM viewer initializes WebGL. Steer ANGLE to desktop GL and disable
# the GPU sandbox / compositing paths (user-provided flags are preserved).
export QTWEBENGINE_CHROMIUM_FLAGS="${QTWEBENGINE_CHROMIUM_FLAGS:---no-sandbox} --disable-gpu-sandbox --disable-gpu-compositing --use-angle=gl"
# If VRM still crashes on your GPU, force software rendering instead:
#   export QTWEBENGINE_CHROMIUM_FLAGS="$QTWEBENGINE_CHROMIUM_FLAGS --disable-gpu"
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
