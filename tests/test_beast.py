import functools
import os
import re
import sys
import tempfile
import warnings
from contextlib import closing
from pathlib import Path
from threading import _shutdown_locks

import packaging.tags
import packaging.version
import pytest

if Path("src").is_dir():
    sys.path.insert(0, "") if "" not in sys.path else None
    lpath, rpath = (sys.path[0 : sys.path.index("") + 1], sys.path[sys.path.index("") + 2 :])
    try:
        sys.path.clear()
        sys.path.__iadd__(lpath + [os.path.join(os.getcwd(), "src")] + rpath)
        import use
    finally:
        sys.path.clear()
        sys.path.__iadd__(lpath + rpath)
import_base = Path(__file__).parent.parent / "src"
is_win = sys.platform.startswith("win")
import use

__package__ = "tests"

import logging

log = logging.getLogger(".".join((__package__, __name__)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)


@pytest.fixture()
def reuse():
    # making a completely new one each time would take ages (_registry)
    use._using = {}
    use._aspects = {}
    use._reloaders = {}
    return use


params = [
    ("olefile", "0.46"),
    ("workerpool", "0.9.4"),
    ("fastcache", "1.1.0"),
    ("readme_renderer", "30.0"),
    ("tiledb", "0.9.5"),
    ("wurlitzer", "3.0.2"),
    ("cctools", "7.0.17"),
    ("clang", "9.0"),
]


@pytest.mark.parametrize("name,version", params)
def test_sample(name, version):
    mod = reuse(name, version=version, modes=reuse.auto_install)
    assert mod
