"""
This is where the story begins. Welcome to JustUse!
Only imports and project-global constants are defined here. 
All superfluous imports are deleted to clean up the namespace - and thus help()

"""


import os
import sys
import tempfile
from datetime import datetime, timezone
from enum import Enum, IntEnum
from logging import basicConfig, getLogger
from pathlib import Path
from uuid import uuid4
from warnings import catch_warnings, filterwarnings

from beartype import beartype
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

sessionID = uuid4()
del uuid4

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))

try:
    home.mkdir(mode=0o755, parents=True, exist_ok=True)
except PermissionError:
    # this should fix the permission issues on android #80
    home = tempfile.mkdtemp(prefix="justuse_")

if sys.version_info < (3, 11):
    import tomli as toml
else:
    import tomllib as toml

(home / "config.toml").touch(mode=0o755, exist_ok=True)

with open(home / "config.toml", "rb") as f:
    config_dict = toml.load(f)
del toml

from use.pydantics import Configuration

config = Configuration(**config_dict)
del Configuration

config.logs.mkdir(mode=0o755, parents=True, exist_ok=True)
config.packages.mkdir(mode=0o755, parents=True, exist_ok=True)

for file in (
    config.logs / "usage.log",
    config.registry,
):
    file.touch(mode=0o755, exist_ok=True)


def fraction_of_day(now: datetime = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
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
    level=config.debug_level,
)

# !!! SEE NOTE !!!
# IMPORTANT; The setup.py script must be able to read the
# current use __version__ variable **AS A STRING LITERAL** from
# this file. If you do anything except updating the version,
# please check that setup.py can still be executed.
__version__ = "0.8.1.1"
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


import hashlib


class Hash(Enum):
    sha256 = hashlib.sha256
    blake = hashlib.blake2s


del hashlib


class Modes(IntEnum):
    auto_install = 2**0
    fatal_exceptions = 2**1
    reloading = 2**2
    no_public_installation = 2**3
    fastfail = 2**4
    recklessness = 2**5
    no_browser = 2**6
    no_cleanup = 2**7


from use.aspectizing import (
    apply_aspect,
    iter_submodules,
    show_aspects,
    show_profiling,
    tinny_profiler,
    woody_logger,
)
from use.buffet import buffet_table
from use.main import URL, ProxyModule, Use, test_version
from use.pydantics import Version, git

for member in Modes:
    setattr(Use, member.name, member.value)

use = Use()

del os
use.__dict__.update(dict(globals()))
use = ProxyModule(use)
if not test_version:
    sys.modules["use"] = use

del test_version
del use.__dict__["test_version"]
del sys
del use.__dict__["sys"]

with catch_warnings():
    filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning, module="beartype")
    apply_aspect(use.iter_submodules(use), beartype)
