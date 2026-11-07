# Qt resource registration for the Linux build of v2.4.7.
#
# The upstream release shipped a generated (binary) resources.py that
# embedded the "sowInterface" resource bundle. In the Linux port the
# bundle is stored next to this module as a compiled resource file
# (sowInterface.rcc, built from sowInterface.qrc with `rcc`) and is
# registered here at import time. main.py imports this module on
# startup, so ":/sowInterface/*.png" paths referenced by the GUI QSS
# resolve correctly.

import os

from PyQt6.QtCore import QResource

_RCC_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sowInterface.rcc"
)

if os.path.exists(_RCC_PATH):
    QResource.registerResource(_RCC_PATH)
