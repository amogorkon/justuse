import subprocess
import sys

from justuse import Path


def test_setup_py_works(reuse):
    with reuse.ScopedCwd(Path(__file__).parent.parent):
        result = subprocess.check_output(
            [sys.executable, "setup.py", "--help"], shell=False
        )
        assert result
