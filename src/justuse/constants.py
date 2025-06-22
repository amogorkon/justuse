"""
Constants, enums, and flags for justuse.
"""

import hashlib
from enum import Enum, Flag, IntEnum, auto


class Hash(Enum):
    sha256 = hashlib.sha256
    blake = hashlib.blake2s


class Modes(IntEnum):
    auto_install = 2**0
    fatal_exceptions = 2**1
    reloading = 2**2
    no_public_installation = 2**3
    fastfail = 2**4
    recklessness = 2**5
    no_browser = 2**6
    no_cleanup = 2**7


class ModeFlags(Flag):
    AUTO_INSTALL = auto()
    FATAL_EXCEPTIONS = auto()
    RELOADING = auto()
    FASTFAIL = auto()
    RECKLESS = auto()
    DEFAULT = auto()
    NO_PUBLIC_INSTALLATION = auto()
    NO_CLEANUP = auto()
    NO_BROWSER = auto()


(
    AUTO_INSTALL,
    FATAL_EXCEPTIONS,
    RELOADING,
    FASTFAIL,
    RECKLESS,
    DEFAULT,
    NO_PUBLIC_INSTALLATION,
    NO_CLEANUP,
    NO_BROWSER,
) = ModeFlags
