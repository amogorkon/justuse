"""
This is where the story begins. Welcome to JustUse!
Only imports and project-global constants are defined here. 
    
"""

import sys
if sys.version_info < (3, 9) and "tests" not in sys.modules:
    import gc, types, typing
    from typing import _GenericAlias as GenericAlias
    for t in (list, dict, set, tuple, frozenset):
      r = gc.get_referents(t.__dict__)[0]
      r.update({
        "__class_getitem__": classmethod(GenericAlias),
      })


import hashlib
import os
from collections import namedtuple
from enum import Enum, IntEnum
from inspect import isfunction, ismethod  # for aspectizing, DO NOT REMOVE
from logging import DEBUG, INFO, NOTSET, getLogger, root
from pathlib import Path

from beartype import beartype

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


class VersionWarning(Warning):
    pass


class NotReloadableWarning(Warning):
    pass


class NoValidationWarning(Warning):
    pass


class AmbiguityWarning(Warning):
    pass


class UnexpectedHash(ImportError):
    pass


class AutoInstallationError(ImportError):
    pass


from use.hash_alphabet import *

ModInUse = namedtuple("ModInUse", "name mod path spec frame")
NoneType = type(None)


class Hash(Enum):
    sha256 = hashlib.sha256


class Modes(IntEnum):
    auto_install = 2 ** 0
    fatal_exceptions = 2 ** 1
    reloading = 2 ** 2
    no_public_installation = 2 ** 4
    fastfail = 2 ** 5


config = {"version_warning": True, "debugging": False, "use_db": True}

if sys.version_info < (3, 10):
    from use.buffet_old import buffet_table
else:
    from use.buffet import buffet_table

from use.main import *
from use.messages import *

### NEEDED FOR TESTS!! ###
from use.pimp import *
from use.pimp import _get_package_data, _get_version, _is_version_satisfied, get_supported
from use.pypi_model import *
from use.tools import *

use = Use()
use.__dict__.update({k: v for k, v in globals().items()})  # to avoid recursion-confusion
use = ProxyModule(use)


def decorator_log_calling_function_and_args(func, *args):
    """
    Decorator to log the calling function and its arguments.

    Args:
        func (function): The function to decorate.
        *args: The arguments to pass to the function.

    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        log.debug(f"{func.__name__}({args}, {kwargs})")
        return func(*args, **kwargs)

    return wrapper


use @ (isfunction, "", beartype)
use @ (isfunction, "", decorator_log_calling_function_and_args)

if not test_version:
    sys.modules["use"] = use
