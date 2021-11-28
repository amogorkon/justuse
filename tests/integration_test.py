import functools
import os
import re
import sys
import tempfile
import warnings
from contextlib import closing
from importlib.machinery import SourceFileLoader
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
from tests.unit_test import reuse, ScopedCwd

import logging

log = logging.getLogger(".".join((__package__, __name__)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)


params = [
    # ("olefile", "0.46"), # Windows-only
    ("workerpool", "0.9.4"),
    ("fastcache", "1.1.0"),
    ("pytest-cov", "2.12.1"),
    ("pytest-env", "0.6.2"),
    ("requests", "2.24.0"),
    ("furl", "2.1.2"),
    ("wheel", "0.36.2"),
    ("icontract", "2.5.4"),
    ("tiledb", "0.9.5"),
    ("wurlitzer", "3.0.2"),
    # ("cctools", "7.0.17"), # too slow, takes minutes to build
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


def test_aspectize_defaults(reuse):
    # baseline
    srcdir = Path(__file__).parent.parent
    if "tests.simple_funcs" in sys.modules:
        del sys.modules["tests.simple_funcs"]
    with ScopedCwd(srcdir):
        mod = reuse(reuse.Path("./tests/simple_funcs.py"), package_name="tests")
        assert mod.two() == 2


def test_aspectize_function_by_name(reuse):
    # functions with specific names only
    srcdir = Path(__file__).parent.parent
    if "tests.simple_funcs" in sys.modules:
        del sys.modules["tests.simple_funcs"]
    with ScopedCwd(srcdir):
        mod = (
            reuse(reuse.Path("./tests/simple_funcs.py"), package_name="tests")
            @ (reuse.isfunction, "two", double_function)
        )
        assert mod.two() == 4
        assert mod.three() == 3
        assert reuse.ismethod


def test_aspectize_all_functions(reuse):
    # all functions, but not classes or methods
    srcdir = Path(__file__).parent.parent
    if "tests.simple_funcs" in sys.modules:
        del sys.modules["tests.simple_funcs"]
    with ScopedCwd(srcdir):
        mod = (
            reuse(reuse.Path("./tests/simple_funcs.py"), package_name="tests")
            @ (reuse.isfunction, "", double_function)
        )
        assert mod.two() == 4
        assert mod.three() == 6
        inst = mod.Two()
        assert inst() == 2
        inst = mod.Three()
        assert inst.three() == 3


def test_simple_url(reuse):
    import http.server

    port = 8089
    orig_cwd = Path.cwd()
    try:
        os.chdir(Path(__file__).parent.parent)

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


def test_autoinstall_numpy_dual_version(reuse):
    ver1, ver2 = "1.19.3", "1.19.5"
    for ver in (ver1, ver2):
        for k,v in list(sys.modules.items()):
            if k == "numpy" or k.startswith("numpy."):
                loader = (
                    getattr(v, "__loader__", None)
                    or v.__spec__.loader
                )
                if isinstance(loader, SourceFileLoader):
                    del sys.modules[k]
        
        mod = suggested_artifact(reuse, "numpy", version=ver)
        assert mod
        assert mod.__version__ == ver
    

def test_autoinstall_protobuf(reuse):
    ver = "3.19.1"
    mod = suggested_artifact(
        reuse, "protobuf/google.protobuf", version=ver
    )
    assert mod.__version__ == ver
    assert mod.__name__ == "google.protobuf"
    assert (
        tuple(Path(mod.__file__).parts[-3:])
        == ("google", "protobuf", "__init__.py")
    )


def suggested_artifact(reuse, *args, **kwargs):
    reuse.pimp._clean_sys_modules(args[0].split("/")[-1].split(".")[0])
    try:
        mod = reuse(
            *args, 
            modes=reuse.auto_install | reuse.Modes.fastfail,
            **kwargs
        )
        return mod
    except RuntimeWarning as rw:
        last_line = str(rw).strip().splitlines()[-1].strip()
        log.info("Usimg last line as suggested artifact: %s", repr(last_line))
        last_line2 = last_line.replace("protobuf", "protobuf/google.protobuf")
        mod = eval(last_line2)
        log.info("suggest artifact returning: %s", mod)
        return mod