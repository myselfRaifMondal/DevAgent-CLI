"""Host platform checks, in one place so tests can switch platform without touching ``os.name``.

Patching ``os.name`` to ``"nt"`` on a POSIX machine makes ``pathlib`` try to build
``WindowsPath`` objects, which fails. Code asks ``is_windows()`` instead, and tests
replace that single function.
"""

from __future__ import annotations

import os


def is_windows() -> bool:
    return os.name == "nt"
