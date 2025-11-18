"""
Constants, enums, and flags for justuse.
"""

import hashlib
import inspect
from enum import Enum, Flag, auto


class Hash(Enum):
    sha256 = hashlib.sha256
    blake = hashlib.blake2s


class Modes(Flag):
    """Flags for justuse runtime modes."""

    auto_install = auto()  # Automatically install missing packages
    fatal_exceptions = auto()  # Raise exceptions instead of handling gracefully
    reloading = auto()  # Enable module reloading
    no_public_installation = auto()  # Disallow public package installation
    failfast = auto()  # Fail fast on errors
    recklessness = auto()  # Allow risky operations
    no_browser = auto()  # Disable browser-based features
    no_cleanup = auto()  # Skip cleanup steps
    verbose = auto()  # Enable verbose output
    include_dunder = auto()  # Include dunder (__) methods/attributes
    testing = auto()  # Enable test mode (disable caching, allow cleanup)
    DEFAULT = auto()  # Default mode


class ALL(Enum):
    methods = inspect.ismethod
    properties = inspect.isdatadescriptor
    functions = inspect.isfunction
    classes = inspect.isclass
