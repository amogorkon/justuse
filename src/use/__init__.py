import hashlib
import os
import sys
from collections import namedtuple
from enum import Enum, IntEnum
from logging import DEBUG, INFO, NOTSET, getLogger, root
from pathlib import Path


root.setLevel(DEBUG)

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python"))).absolute()
# !!! SEE NOTE !!!
# IMPORTANT; The setup.py script must be able to read the
# current use __version__ variable **AS A STRING LITERAL** from
# this file. If you do anything except updating the version,
# please check that setup.py can still be executed.
__version__ = "0.6.0"  # IMPORTANT; Must leave exactly as-is for setup
__name__ = "use"
__package__ = "use"
_reloaders: dict["ProxyModule", "ModuleReloader"] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

from use.exceptions import *
from use.hash_alphabet import *

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
### NEEDED FOR TESTS!! ###
import inspect

if sys.version_info < (3, 10):
    from use.buffet_old import buffet_table
else:
    from use.buffet import buffet_table

from use.main import *
from use.messages import *

### NEEDED FOR TESTS!! ###
from use.pimp import *
from use.pimp import _get_package_data, _get_version, _is_version_satisfied, get_supported
from use.platformtag import *
from use.pypi_model import *
from use.tools import *

for k, v in inspect.getmembers(use):
    setattr(sys.modules["use"], k, v)
