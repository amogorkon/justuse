"""
This is where the story begins. Welcome to JustUse!
Only imports and project-global constants are defined here. 
"""


import sys

if sys.version_info < (3, 9) and "tests" not in sys.modules:
    import gc
    import types
    import typing
    from abc import ABCMeta
    from collections.abc import Callable
    from types import CellType
    from typing import _GenericAlias as GenericAlias

    for t in (list, dict, set, tuple, frozenset, ABCMeta, Callable, CellType):
        r = gc.get_referents(t.__dict__)[0]
        r.update(
            {
                "__class_getitem__": classmethod(GenericAlias),
            }
        )

import hashlib
import os
import tempfile
from collections import namedtuple
from datetime import datetime
from enum import Enum, IntEnum
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4
from warnings import catch_warnings, filterwarnings, simplefilter

import toml
from beartype import beartype
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

from use.pydantics import Configuration

sessionID = uuid4()

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))

try:
    home.mkdir(mode=0o755, parents=True, exist_ok=True)
except PermissionError:
    # this should fix the permission issues on android #80
    home = tempfile.mkdtemp(prefix="justuse_")

(home / "config.toml").touch(mode=0o755, exist_ok=True)
config_dict = toml.load(home / "config.toml")

config = Configuration(**config_dict)

config.logs.mkdir(mode=0o755, parents=True, exist_ok=True)
config.packages.mkdir(mode=0o755, parents=True, exist_ok=True)

for file in (
    config.logs / "usage.log",
    config.registry,
):
    file.touch(mode=0o755, exist_ok=True)


def fraction_of_day(now: datetime = None) -> float:
    if now is None:
        now = datetime.utcnow()
    return round(
        (
            now.hour / 24
            + now.minute / (24 * 60)
            + now.second / (24 * 60 * 60)
            + now.microsecond / (24 * 60 * 60 * 1000 * 1000)
        )
        * 1000,
        6,
    )


basicConfig(
    filename=config.logs / "usage.log",
    filemode="a",
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt=f"%Y%m%d {fraction_of_day()}",
    # datefmt="%Y-%m-%d %H:%M:%S",
    level=DEBUG,
)

# !!! SEE NOTE !!!
# IMPORTANT; The setup.py script must be able to read the
# current use __version__ variable **AS A STRING LITERAL** from
# this file. If you do anything except updating the version,
# please check that setup.py can still be executed.
__version__ = "0.7.0"
# for tests
__version__ = os.getenv("USE_VERSION", __version__)
__name__ = "use"
__package__ = "use"

log = getLogger(__name__)
log.info(f"### SESSION START {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} {sessionID} ###")

import use.logutil


class JustuseIssue:
    pass


class NirvanaWarning(Warning, JustuseIssue):
    pass


class VersionWarning(Warning, JustuseIssue):
    pass


class NotReloadableWarning(Warning, JustuseIssue):
    pass


class NoValidationWarning(Warning, JustuseIssue):
    pass


class AmbiguityWarning(Warning, JustuseIssue):
    pass


class UnexpectedHash(ImportError, JustuseIssue):
    pass


class InstallationError(ImportError, JustuseIssue):
    pass


# Coerce all PEP 585 deprecation warnings into fatal exceptions.
with catch_warnings():
    from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

    filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning, module="beartype")

    from use.hash_alphabet import *

    ModInUse = namedtuple("ModInUse", "name mod path spec frame")

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


from use.aspectizing import *
from use.aspectizing import apply_aspect
from use.buffet_old import buffet_table
from use.main import *
from use.messages import *

### NEEDED FOR TESTS!! ###
from use.pimp import *
from use.pimp import _get_data_from_pypi, _get_version, _is_version_satisfied, _parse_name, get_supported
from use.pydantics import *
from use.tools import *

for member in Modes:
    setattr(Use, member.name, member.value)

use = Use()

use.__dict__.update(dict(globals()))
use = ProxyModule(use)

with catch_warnings():
    filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning, module="beartype")
    # apply_aspect(use, beartype, check=isbeartypeable, pattern="")

if not test_version:
    sys.modules["use"] = use
