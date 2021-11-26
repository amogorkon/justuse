"""Here live all the tests that are expected to fail because their functionality is not implemented yet.
Test-Driven Development is done in the following order: 
    1. Create a test that fails.
    2. Write the code that makes the test pass.
    3. Check how long the test took to run. 
    4. If it took longer than 1 second, move it to integration tests. Otherwise, move it to unit tests.
"""

import os
import sys

from contextlib import AbstractContextManager, closing
from hashlib import sha256
from pathlib import Path
from warnings import catch_warnings,  filterwarnings, simplefilter

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st
import use
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pypi_model import JustUse_Info, PyPI_Project, PyPI_Release, Version

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")


@pytest.fixture()
def reuse():
    # making a completely new one each time would take ages (_registry)
    use._using = {}
    use._aspects = {}
    use._reloaders = {}
    return use


    

def test_pypi_model():

    release = PyPI_Release(
        comment_text="test",
        digests={"md5": "asdf"},
        url="https://files.pythonhost",
        ext=".whl",
        packagetype="bdist_wheel",
        distribution="numpy",
        requires_python=False,
        python_version="cp3",
        python_tag="cp3",
        platform_tag="cp4",
        filename="numpy-1.19.5-cp3-cp3-cp4-bdist_wheel.whl",
        abi_tag="cp3",
        yanked=False,
        version="1.19.5",
    )
    assert type(release)(**release.dict()) == release

    info = JustUse_Info(
        distribution="numpy",
        version="1.19.5",
        build_tag="cp4",
        python_tag="cp4",
        abi_tag="cp4",
        platform_tag="cp4",
        ext="whl",
    )
    assert type(info)(**info.dict()) == info


class ScopedCwd(AbstractContextManager):
    def __init__(self, newcwd: Path):
        self._oldcwd = Path.cwd()
        self._newcwd = newcwd

    def __enter__(self, *_):
        os.chdir(self._newcwd)

    def __exit__(self, *_):
        os.chdir(self._oldcwd)


@pytest.mark.skipif(is_win or Path.cwd().as_posix().startswith("/media/"), reason="Not enough hours in a day")
def test_setup_py_works(reuse):
    import subprocess
    with ScopedCwd(Path(__file__).parent.parent):
        result = subprocess.check_output(
            [sys.executable, "setup.py", "--help"],
            shell=False
        )
        assert result


@pytest.mark.skipif(not is_win or not Path.cwd().as_posix().startswith("/media/"), reason="Not enough hours in a day")
@given(st.text())
def test_jack(text):
    assume(text.isprintable())
    sha = sha256(text.encode("utf-8")).hexdigest()
    assert sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))



