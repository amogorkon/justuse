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

from hypothesis import assume, example, given
from hypothesis import strategies as st
from pytest import fixture, mark, skip

orig_cwd = Path().absolute()
os.chdir((Path(__file__).parent.absolute() / "../src").resolve())

import use
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pimp import _parse_name
from use.pydantics import JustUse_Info, PyPI_Project, PyPI_Release, Version

os.chdir(orig_cwd)

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")


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


def test_test(reuse):
    assert reuse
