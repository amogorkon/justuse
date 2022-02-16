"""Here live all the tests that are expected to fail because their functionality is not implemented yet.
Test-Driven Development is done in the following order: 
    1. Create a test that fails.
    2. Write the code that makes the test pass.
    3. Check how long the test took to run. 
    4. If it took longer than 1 second, move it to integration tests. Otherwise, move it to unit tests.
"""

import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, closing
from hashlib import sha256
from pathlib import Path
from warnings import catch_warnings, filterwarnings, simplefilter

import pytest
import use
from hypothesis import assume, example, given
from hypothesis import strategies as st
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pypi_model import JustUse_Info, PyPI_Project, PyPI_Release, Version

from tests.unit_test import ScopedCwd, reuse

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")


def test_443_py_test(reuse):
    try:
        imported = "py" in sys.modules
        import py.test

        if not imported:
            del sys.modules["py"]
    except ImportError:
        pytest.skip("py.test is not installed")
        return
    mod = use("py.test")
    assert mod


def test_441_discord(reuse):
    try:
        imported = "discord" in sys.modules
        import discord

        if not imported:
            del sys.modules["discord"]
    except ImportError:
        pytest.skip("discord is not installed")
        return
    mod = use("discord")
    assert mod


def test_441_discord(reuse):
    try:
        imported = "discord" in sys.modules
        import discord

        if not imported:
            del sys.modules["discord"]
    except ImportError:
        pytest.skip("discord is not installed")
        return
    mod = use("discord")
    assert mod


def test_444_discordpy_autoinstall(reuse):
    mod = use("discord.py", modes=use.auto_install | use.no_browser)
