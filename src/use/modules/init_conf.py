import sys
import os
from collections import namedtuple
from enum import Enum
from importlib.machinery import ModuleSpec, SourceFileLoader
from logging import getLogger, DEBUG, INFO, NOTSET, StreamHandler, root
from typing import Any
from pathlib import Path


log = getLogger(__name__)


# injected via initial_globals for testing, you can safely ignore this
test_config: str = locals().get("test_config", {})
test_version: str = locals().get("test_version", None)
__name__ = "use.use"
__package__ = "use"
__path__ = __file__
__spec__ = ModuleSpec("use.use", loader=SourceFileLoader(fullname="use", path=__file__))
__spec__.submodule_search_locations = [Path(__file__).parent]
__version__ = test_version or "0.5.0"

_reloaders: dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

ModInUse = namedtuple("ModInUse", "name mod path spec frame")
NoneType = type(None)


# Really looking forward to actual builtin sentinel values..
mode = Enum("Mode", "fastfail")

config = {"version_warning": True, "debugging": False, "use_db": True}


# initialize logging
root.addHandler(StreamHandler(sys.stderr))
root.setLevel(NOTSET)
if "DEBUG" in os.environ or "pytest" in getattr(
    sys.modules.get("__init__", ""), "__file__", ""
):
    root.setLevel(DEBUG)
    test_config["debugging"] = True
else:
    root.setLevel(INFO)


use = sys.modules.get(__name__)
