import io
import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, closing, redirect_stdout
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from subprocess import STDOUT, check_output
from textwrap import dedent
from threading import _shutdown_locks
from time import time
from types import ModuleType
from unittest.mock import patch
from warnings import catch_warnings, filterwarnings

import packaging.tags
import packaging.version
import pytest
import requests
from furl import furl as URL
from hypothesis import assume, example, given
from hypothesis import strategies as st
from pytest import fixture, mark, raises, skip

src = import_base = Path(__file__).parent.parent / "src"
cwd = Path().cwd()
os.chdir(src)
sys.path.insert(0, "") if "" not in sys.path else None

if sys.version_info < (3, 9) and "use" not in sys.modules:
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


import use

os.chdir(cwd)

is_win = sys.platform.startswith("win")

__package__ = "tests"
import json
import logging

from use import auto_install, fatal_exceptions, no_cleanup, use
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pimp import _parse_name
from use.pydantics import JustUse_Info, PyPI_Project, PyPI_Release, Version

log = logging.getLogger(".".join((__package__, __name__)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)

use.config.testing = True


@fixture()
def reuse():
    """
    Return the `use` module in a clean state for "reuse."

    NOTE: making a completely new one each time would take
    ages, due to expensive _registry setup, re-downloading
    venv packages, etc., so if we are careful to reset any
    additional state changes on a case-by-case basis,
    this approach is more efficient and is the clear winner.
    """
    use._using.clear()
    use.main._reloaders.clear()
    return use


p = Path(__file__).parent / "beast_data.json"

with open(p) as file:
    data = json.load(file)

begin = time()


@mark.parametrize("package_name, module_name, version", data)
def test_mass(reuse, package_name, module_name, version):
    """
    Taken from the original beast test suite, and modified boiled down to the bare minimum.
    """
    with patch("webbrowser.open"), io.StringIO() as buf, redirect_stdout(buf):
        try:
            mod = reuse(package_name, version=version, modes=reuse.auto_install)
            assert False, f"Actually returned mod: {mod}"
        except RuntimeWarning:
            recommended_hash = buf.getvalue().splitlines()[-1].strip()
        mod = reuse(
            package_name=package_name,
            module_name=module_name,
            version=version,
            hashes=recommended_hash,
            modes=reuse.auto_install | reuse.no_cleanup,
        )
        assert isinstance(mod, ModuleType)


print("============================================================")
print("ran", len(data), "tests in", (time() - begin) // 60, "minutes")
