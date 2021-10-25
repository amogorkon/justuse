import logging
import os
import re
import shlex
import sys
from typing import Iterator
from pathlib import Path

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger("air")


try:
    import warnings

    try:
        from beartype.roar._roarwarn import BeartypeDecorHintPepWarning

        warnings.filterwarnings(action="ignore", category=AnnoyingBeartypeWarning)
    except (NameError, ImportError):
        pass
except ImportError:
    pass

__package__ = "use"
home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python"))).absolute()
# !!! SEE NOTE !!!
# IMPORTANT; The setup.py script must be able to read the
# current use __version__ variable **AS A STRING LITERAL** from
# this file. If you do anything except updating the version,
# please check that setup.py can still be executed.
__version__ = "0.6.0"  # IMPORTANT; Must leave exactly as-is for setup
# !!! SEE NOTE !!!

import sys
import os
from collections import namedtuple
from enum import Enum, IntEnum
from importlib.machinery import ModuleSpec, SourceFileLoader
from logging import getLogger, DEBUG, INFO, NOTSET, StreamHandler, root
from typing import Any
from pathlib import Path


log = getLogger(__name__)


# injected via initial_globals for testing, you can safely ignore this
__name__ = "use"
__package__ = "use"

_reloaders: dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

ModInUse = namedtuple("ModInUse", "name mod path spec frame")
NoneType = type(None)


# Really looking forward to actual builtin sentinel values..
class Modes(IntEnum):
    auto_install = 2 ** 0
    fatal_exceptions = 2 ** 1
    reloading = 2 ** 2
    no_public_installation = 2 ** 4
    fastfail = 2 ** 5


config = {"version_warning": True, "debugging": False, "use_db": True}


# initialize logging
root.addHandler(StreamHandler(sys.stderr))
root.setLevel(NOTSET)
if "DEBUG" in os.environ or "pytest" in getattr(sys.modules.get("__init__", ""), "__file__", ""):
    root.setLevel(DEBUG)
else:
    root.setLevel(INFO)


print("Startong submod imports: use.hash_alphabet")
from use.hash_alphabet import *

print("Startong submod imports: use.modules")
from use.modules.Decorators import *
from use.modules.Hashish import *
from use.modules.PlatformTag import *
from use.modules.install_utils import *

print("Startong submod imports: use.pypi_model")
from use.pypi_model import *

print("Startong submod imports: use.modules.Messages")
from use.modules.Messages import *

print("Startong submod imports: use.use")
from use.use import *
import inspect

print("Finished importing modules")
for k, v in inspect.getmembers(use):
    setattr(sys.modules["use"], k, v)
