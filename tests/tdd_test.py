"""Here live all the tests that are expected to fail because their functionality is not implemented yet.
Test-Driven Development is done in the following order: 
    1. Create a test that fails.
    2. Write the code that makes the test pass.
    3. Check how long the test took to run. 
    4. If it took longer than 1 second, move it to integration tests. Otherwise, move it to unit tests.
"""

import os
import sys

from hypothesis import strategies as st
from pytest import fixture, mark, raises, skip

is_win = sys.platform.startswith("win")

__package__ = "tests"
import logging

from use import auto_install, fatal_exceptions, no_cleanup, use

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


def test_test(reuse):
    assert reuse
