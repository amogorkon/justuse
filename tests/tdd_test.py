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

import use
from hypothesis import assume, example, given
from hypothesis import strategies as st
from pytest import mark, skip
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pimp import _parse_name
from use.pypi_model import JustUse_Info, PyPI_Project, PyPI_Release, Version

from tests.unit_test import ScopedCwd, reuse

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")


def test_51_sqlalchemy(reuse):
    mod = use(
        "sqlalchemy",
        version="0.7.1",
        hashes={"5ef95d19c31a8cd3905c697be0a7e94e70ab1926ecd4159c3e6c1cf01fc3c492"},
        # SQLAlchemy-0.7.1.tar.gz (2.3 MB) - only a single artifact
        # Uploaded Jun 5, 2011 source
        modes=use.auto_install,
    )
    assert mod
