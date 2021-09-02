import functools
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from threading import _shutdown_locks

import packaging.tags
import packaging.version
import pytest
from furl import furl as URL

# this is actually a test!
from tests.simple_funcs import three

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


def test_access_to_home(reuse):
    test = reuse.Path.home() / ".justuse-python/packages/test"
    test.touch(mode=0o644, exist_ok=True)
    with open(test, "w") as file:
        file.write("test")
    assert test.exists()
    test.unlink()
    assert not test.exists()


def test_other_case(reuse):
    with pytest.raises(NotImplementedError):
        reuse(2, modes=reuse.fatal_exceptions)


def test_fail_dir(reuse):
    with pytest.raises(ImportError):
        reuse(Path(""))


def test_simple_path(reuse):
    foo_path = Path(".tests/foo.py")
    print(f"loading foo module via use(Path('{foo_path}'))")
    mod = reuse(Path(foo_path), initial_globals={"a": 42})
    assert mod.test() == 42


def test_simple_url(reuse):
    import http.server

    port = 8089
    with http.server.HTTPServer(("", port), http.server.SimpleHTTPRequestHandler) as svr:
        foo_uri = f"http://localhost:{port}/tests/.tests/foo.py"
        print(f"starting thread to handle HTTP request on port {port}")
        import threading

        thd = threading.Thread(target=svr.handle_request)
        thd.start()
        print(f"loading foo module via use(URL({foo_uri}))")
        with pytest.warns(use.NoValidationWarning):
            mod = reuse(URL(foo_uri), initial_globals={"a": 42})
            assert mod.test() == 42


def test_internet_url(reuse):
    foo_uri = "https://raw.githubusercontent.com/greyblue9/justuse/3f783e6781d810780a4bbd2a76efdee938dde704/tests/foo.py"
    print(f"loading foo module via use(URL({foo_uri}))")
    mod = reuse(
        URL(foo_uri),
        initial_globals={"a": 42},
        hash_algo=use.Hash.sha256,
        hash_value="b136efa1d0dab3caaeb68bc41258525533d9058aa925d3c0c5e98ca61200674d",
    )
    assert mod.test() == 42


def test_module_package_ambiguity(reuse):
    original_cwd = os.getcwd()
    os.chdir(Path("tests/.tests"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reuse("sys", modes=reuse.fatal_exceptions)
    w_filtered = [*filter(lambda i: i.category is not DeprecationWarning, w)]
    assert len(w_filtered) == 1
    assert issubclass(w_filtered[-1].category, use.AmbiguityWarning)
    assert "local module" in str(w_filtered[-1].message)
    os.chdir(original_cwd)


def test_builtin():
    # must be the original use because loading builtins requires looking up _using, which mustn't be wiped for this reason
    mod = use("sys")
    assert mod.path is sys.path


def test_classical_install(reuse):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("pytest", modes=reuse.fatal_exceptions)
        assert mod is pytest or mod._ProxyModule__implementation is pytest
        assert issubclass(w[-1].category, use.AmbiguityWarning)


def test_autoinstall_PEBKAC(reuse):
    # auto-install requested, but no version or hashes specified
    with pytest.raises(RuntimeWarning):
        reuse("pytest", modes=reuse.auto_install)

    # forgot hashes
    with pytest.raises(packaging.version.InvalidVersion):
        reuse("pytest", version="-1", modes=reuse.auto_install)

    # forgot version
    with pytest.raises(RuntimeWarning):
        reuse(
            "pytest",
            hashes="asdf",
            modes=reuse.auto_install,
        )

    # impossible version
    with pytest.raises(AssertionError):  # version must be either str or tuple
        reuse(
            "pytest",
            modes=reuse.auto_install,
            version=-1,
            hashes="asdf",
        )

    # non-existing package
    with pytest.raises(ImportError):
        reuse(
            "4-^df",
            modes=reuse.auto_install,
            version="0.0.1",
            hashes="asdf",
        )


def test_version_warning(reuse):
    # no auto-install requested, wrong version only gives a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reuse("pytest", version="0.0", modes=reuse.fatal_exceptions)
        assert issubclass(w[-1].category, (use.AmbiguityWarning, use.VersionWarning))


def suggested_artifact(*args, **kwargs):
    import use

    reuse = use
    rw = None
    try:
        mod = reuse(*args, modes=reuse.auto_install | reuse.fatal_exceptions, **kwargs)
    except RuntimeWarning as r:
        rw = r
    except BaseException as e:
        raise AssertionError(
            f"suggested_artifact failed for use("
            f"{', '.join(map(repr, args))}, "
            f"{', '.join(map(repr, kwargs.items()))}"
            f"): {e}"
        ) from e
    assert rw
    assert "version=" in str(rw), f"warning does not suggest a version: {rw}"
    assert "hashes=" in str(rw), f"warning does not suggest a hash: {rw}"
    assert isinstance(rw.args[0], str)
    match = re.search(
        'version="?(?P<version>[^"]+)".*' "hashes=?(?P<hashes>[^()]+), ",
        str(rw),
    )
    assert match
    hashes_evalstr = match.group("hashes")
    log.debug("eval'ing the following string from rw message: %r", hashes_evalstr)
    hashes = eval(hashes_evalstr)
    log.debug("eval'ed to the following value: %r", hashes)
    assert isinstance(
        hashes, set
    ), f"The wrong type of object is given in the warning message: {rw}"
    version = match.group("version")
    return (version, hashes)


def test_use_global_install(reuse):
    from . import foo

    with pytest.raises(NameError):
        foo.bar()

    reuse.install()
    assert foo.bar()
    reuse.uninstall()
    del foo


def test_is_version_satisfied(reuse):
    sys_version = packaging.version.Version("3.6.0")
    # google.protobuf 1.19.5 normal case
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "cp36",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert reuse._is_version_satisfied(info, sys_version)

    # requires >= python 4!
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "cp36",
        "requires_python": ">=4",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert not reuse._is_version_satisfied(info, sys_version)

    # pure python
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": "google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "source",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert reuse._is_version_satisfied(info, sys_version)


def test_find_windows_artifact(reuse):
    data = reuse._get_package_data("protobuf")
    assert "3.17.3" in data["releases"]


def test_parse_filename(reuse):
    assert reuse._parse_filename("protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl") == {
        "distribution": "protobuf",
        "version": "1.19.5",
        "build_tag": None,
        "python_tag": "cp36",
        "abi_tag": "cp36m",
        "platform_tag": "macosx_10_9_x86_64",
        "ext": "whl",
    }


def test_classic_import_no_version(reuse):
    rw = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            reuse("furl", modes=reuse.auto_install)
            assert issubclass(w[-1].category, reuse.AmbiguityWarning)
            return
        except RuntimeWarning as w:
            rw = w
    log.warning(f"from try/catch: {rw=}")


def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("furl", version=version, modes=reuse.fatal_exceptions)
        assert not w
        assert reuse.Version(mod.__version__) == version


def test_classic_import_diff_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        major, minor, patch = version
        mod = reuse(
            "furl",
            version=reuse.Version(major=major, minor=minor, patch=patch + 1),
            modes=reuse.fatal_exceptions,
        )
        assert issubclass(w[-1].category, reuse.VersionWarning)
        assert reuse.Version(mod.__version__) == version


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="code lines can't be looked up? # TODO"
)
def test_use_ugrade_version_warning(reuse):
    version = "0.0.0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # no other way to change __version__ before the actual import while the version check happens on import
        test_use = reuse(
            reuse.Path(r"../src/use/use.py"), initial_globals={"test_version": version}
        )
        assert test_use.test_version == test_use.__version__ == version
        assert w[0].category.__name__ == reuse.VersionWarning.__name__


class Restorer:
    def __enter__(self):
        self.locks = set(_shutdown_locks)

    def __exit__(self, arg1, arg2, arg3):
        for lock in set(_shutdown_locks).difference(self.locks):
            lock.release()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="windows reloading")
def test_reloading(reuse):
    fd, file = tempfile.mkstemp(".py", "test_module")
    with Restorer():
        mod = None
        newfile = f"{file}.t"
        for check in range(1, 5):
            with open(newfile, "w") as f:
                f.write(f"def foo(): return {check}")
                f.flush()
            os.rename(newfile, file)
            mod = mod or reuse(Path(file), modes=reuse.reloading)
            while mod.foo() < check:
                pass


@pytest.mark.skipif(sys.platform.startswith("win"), reason="windows reloading")
def test_suggestion_works(reuse):
    sugg = suggested_artifact("xdis")
    mod = reuse(
        "xdis",
        version=sugg[0],
        hashes=sugg[1],
        modes=use.auto_install
    )
    assert mod

def double_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) * 2

    return wrapper


def test_aspectize(reuse):  # sourcery skip: extract-duplicate-method
    # baseline
    mod = reuse(reuse.Path("simple_funcs.py"))
    assert mod.two() == 2

    # all functions, but not classes or methods
    mod = reuse(
        reuse.Path("simple_funcs.py"),
        aspectize={(reuse.isfunction, ""): double_function},
    )

    assert mod.two() == 4
    assert mod.three() == 6
    inst = mod.Two()
    assert inst() == 2
    inst = mod.Three()
    assert inst.three() == 3

    # functions with specific names only
    mod = reuse(
        reuse.Path("simple_funcs.py"),
        aspectize={(reuse.isfunction, "two"): double_function},
    )
    assert mod.two() == 4
    assert mod.three() == 3
    assert reuse.ismethod