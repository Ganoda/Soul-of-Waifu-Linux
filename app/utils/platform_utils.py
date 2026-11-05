"""
Small cross-platform utility helpers shared across the app.
"""

import os
import shutil
import subprocess
import sys
import logging

logger = logging.getLogger("PlatformUtils")

# Preferred openers on Linux/BSD, tried in order (quiet fallback chain).
_POSIX_OPENERS = ("xdg-open", "gio", "kde-open", "gnome-open")


def open_path(path) -> bool:
    """
    Open a file or folder with the platform's default handler.

    win32 -> os.startfile, macOS -> 'open', Linux/BSD -> xdg-open
    (with quiet fallbacks to other common openers when xdg-open is absent).

    Returns True if an opener was successfully spawned, False otherwise.
    """
    target = str(path)
    try:
        if sys.platform == "win32":
            os.startfile(target)
            return True

        if sys.platform == "darwin":
            subprocess.Popen(["open", target])
            return True

        # Linux/BSD: try the preferred openers in order, fail quietly if none exist.
        for opener in _POSIX_OPENERS:
            opener_exe = shutil.which(opener)
            if opener_exe:
                subprocess.Popen([opener_exe, target])
                return True

        logger.warning(f"No suitable opener found to open '{target}' (tried {_POSIX_OPENERS}).")
        return False
    except Exception as e:
        logger.warning(f"Failed to open '{target}': {e}")
        return False
