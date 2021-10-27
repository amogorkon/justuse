import hashlib
import logging
import os
import sys
from collections import namedtuple
from enum import Enum, IntEnum
from importlib.machinery import ModuleSpec, SourceFileLoader
from logging import DEBUG, INFO, NOTSET, StreamHandler, getLogger, root
from pathlib import Path
from typing import Any, Iterator

logging.root.setLevel(logging.DEBUG)
logger = logging.getLogger("air")


try:
    import warnings

    try:
        from beartype.roar._roarwarn import BeartypeDecorHintPepWarning

        warnings.filterwarnings(action="ignore", category=BeartypeDecorHintPepWarning)
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


log = getLogger(__name__)
__name__ = "use"
__package__ = "use"

_reloaders: dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

ModInUse = namedtuple("ModInUse", "name mod path spec frame")
NoneType = type(None)


class Hash(Enum):
    sha256 = hashlib.sha256


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
from use.pimp import *
from use.platformtag import *
from use.tools import *

print("Startong submod imports: use.pypi_model")
from use.pypi_model import *

print("Starting submod imports: use.modules.Messages")
from use.messages import *

print("Startong submod imports: use.use")
import inspect

from use.main import *
### NEEDED FOR TESTS!! ###
from use.pimp import _get_package_data, _get_version, _is_version_satisfied, get_supported

print("Finished importing modules")
for k, v in inspect.getmembers(use):
    setattr(sys.modules["use"], k, v)
