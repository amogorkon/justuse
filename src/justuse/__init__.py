# Version string for packaging and runtime
__version__ = "2025.26"

from pathlib import Path

from .config import home, sessionID
from .constants import Hash, Modes
from .main import URL, ProxyModule, Use
from .repo import Repo

auto_install = Modes.auto_install
no_cleanup = Modes.no_cleanup
reloading = Modes.reloading

use = ProxyModule(Use())

__all__ = [
    "home",
    "sessionID",
    "Hash",
    "Modes",
    "auto_install",
    "no_cleanup",
    "reloading",
    "ProxyModule",
    "use",
    "Path",
    "URL",
    "Repo",
]
