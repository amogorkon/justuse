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
def test_sample(reuse, name, version):
    try:
        reuse(name, version=version, modes=reuse.auto_install)
    except BaseException as ie:
        suggestion = ie.args[0].strip().splitlines()[-1]
        log.debug("suggestion = %s", repr(suggestion))
        mod = eval(suggestion)
        assert mod
        return
    assert False, "Should raise ImportError: missing hashes."

@pytest.mark.parametrize("name, version", (("numpy", "1.19.3"),))
def test_86_numpy(reuse, name, version):
    use = reuse  # for the eval() later
    with pytest.raises(RuntimeWarning) as w:
        reuse(name, version=version, modes=reuse.auto_install)
    assert w
    recommendation = str(w.value).split("\n")[-1].strip()
    mod = eval(recommendation)
    assert mod.__name__ == reuse._parse_name(name)[1]
    return mod  # for the redownload test


@pytest.mark.skipif(True, reason="Needs investigation")
def test_redownload_module(reuse):
    def inject_fault(*, path, **kwargs):
        log.info("fault_inject: deleting %s", path)
        path.delete()

    assert test_86_numpy(reuse, "example-pypi-package/examplepy", "0.1.0")
    try:
        reuse.config["fault_inject"] = inject_fault
        assert test_86_numpy(reuse, "example-pypi-package/examplepy", "0.1.0")
    finally:
        del reuse.config["fault_inject"]
    assert test_86_numpy(reuse, "example-pypi-package/examplepy", "0.1.0")


def double_function(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) * 2

    return wrapper


def test_aspectize(reuse):
    # baseline
    mod = reuse(reuse.Path("simple_funcs.py"))
    assert mod.two() == 2

    # all functions, but not classes or methods
    mod = reuse(reuse.Path("simple_funcs.py")) @ (reuse.isfunction, "", double_function)

    assert mod.two() == 4
    assert mod.three() == 6
    inst = mod.Two()
    assert inst() == 2
    inst = mod.Three()
    assert inst.three() == 3

    # functions with specific names only
    mod = reuse(reuse.Path("simple_funcs.py")) @ (reuse.isfunction, "two", double_function)
    assert mod.two() == 4
    assert mod.three() == 3
    assert reuse.ismethod


def test_simple_url(reuse):
    import http.server

    port = 8089
    orig_cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).parent.parent.parent)

        with http.server.HTTPServer(("", port), http.server.SimpleHTTPRequestHandler) as svr:
            foo_uri = f"http://localhost:{port}/tests/.tests/foo.py"
            print(f"starting thread to handle HTTP request on port {port}")
            import threading

            thd = threading.Thread(target=svr.handle_request)
            thd.start()
            print(f"loading foo module via use(URL({foo_uri}))")
            with pytest.warns(use.NoValidationWarning):
                mod = reuse(reuse.URL(foo_uri), initial_globals={"a": 42})
                assert mod.test() == 42
    finally:
        os.chdir(orig_cwd)