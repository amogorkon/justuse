import os
import re
import runpy
import subprocess
import sys
import tempfile
import warnings
from contextlib import AbstractContextManager, closing
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from threading import _shutdown_locks
from unittest.mock import patch

import packaging.tags
import packaging.version
import pytest
from furl import furl as URL
from hypothesis import assume, example, given
from hypothesis import strategies as st


is_win = sys.platform.startswith("win")

__package__ = "tests"
import use
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest
import logging

log = logging.getLogger(".".join((__package__, __name__)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)

use.config["testing"] = True

# this is actually a test!
from tests.simple_funcs import three


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
    foo_path = Path(__file__).parent / ".tests" / "foo.py"
    print(f"loading foo module via use(Path('{foo_path}'))")
    mod = reuse(Path(foo_path), initial_globals={"a": 42})
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
    try:
        os.chdir(Path(__file__).parent / ".tests")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            reuse("sys", modes=reuse.fatal_exceptions)
        w_filtered = [*filter(lambda i: i.category is not DeprecationWarning, w)]
        assert len(w_filtered) == 1
        assert issubclass(w_filtered[-1].category, use.AmbiguityWarning)
        assert "local module" in str(w_filtered[-1].message)
    finally:
        os.chdir(original_cwd)


def test_builtin():
    # must be the original use because loading builtins requires looking up _using, which mustn't be wiped for this reason
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = use("sys")
        assert mod.path is sys.path


def test_classical_install(reuse):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("pytest", version=pytest.__version__, modes=reuse.fatal_exceptions)
        assert mod is pytest or mod._ProxyModule__implementation is pytest
        assert not w


def test_classical_install_no_version(reuse):
    mod = reuse("pytest")
    assert mod is pytest or mod._ProxyModule__implementation is pytest


def test_PEBKAC_hash_no_version(reuse):
    with pytest.raises(RuntimeWarning):
        reuse(
            "pytest",
            hashes="asdf",
            modes=reuse.auto_install,
        )


def test_PEBKAC_nonexisting_pkg(reuse):
    # non-existing pkg
    with pytest.raises(ImportError):
        reuse(
            "4-^df",
            modes=reuse.auto_install,
            version="0.0.1",
            hashes="asdf",
        )


def test_PEBKAC_impossible_version(reuse):
    # impossible version
    with pytest.raises(TypeError):  # version must be either str or tuple
        reuse(
            "pytest",
            modes=reuse.auto_install,
            version=-1,
            hashes="asdf",
        )


def test_autoinstall_PEBKAC(reuse):
    with patch("webbrowser.open"):
        # auto-install requested, but no version or hashes specified
        with pytest.raises(RuntimeWarning):
            reuse("pytest", modes=reuse.auto_install)

        # forgot hashes
        with pytest.raises(packaging.version.InvalidVersion):
            reuse("pytest", version="-1", modes=reuse.auto_install)


def test_version_warning(reuse):
    # no auto-install requested, wrong version only gives a warning
    try:
        reuse("pytest", version="0.0", modes=reuse.fatal_exceptions)
    except use.AmbiguityWarning:
        pass


def test_use_global_install(reuse):
    from . import foo

    with pytest.raises(NameError):
        foo.bar()

    reuse.install()
    assert foo.bar()
    reuse.uninstall()
    del foo


def test_is_version_satisfied(reuse):
    sys_version = reuse.Version("3.6.0")
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
    assert reuse._is_version_satisfied(info.get("requires_python", ""), sys_version)

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
    assert False == reuse._is_version_satisfied(info.get("requires_python", ""), reuse.Version(sys_version))

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
    assert reuse._is_version_satisfied(info.get("requires_python", ""), sys_version)


def test_find_windows_artifact(reuse):
    data = reuse._get_package_data("protobuf")
    assert reuse.Version("3.17.3") in reuse._get_package_data("protobuf").releases


def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("furl", version=version)
        assert not w
        assert reuse.Version(mod.__version__) == reuse.Version(version)


def test_classic_import_diff_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    try:
        major, minor, patch = version
        mod = reuse(
            "furl",
            version=reuse.Version(major=major, minor=minor, patch=patch + 1),
            modes=reuse.fatal_exceptions,
        )
    except use.AmbiguityWarning:
        pass


@pytest.mark.skipif(True, reason="Not working, needs investigation")
def test_use_ugrade_version_warning(reuse):
    version = "0.0.0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # no other way to change __version__ before the actual import while the version check happens on import
        test_use = reuse(
            reuse.Path(reuse.__file__).absolute(),
            initial_globals={
                "test_version": version,
                "test_config": {"version_warning": True},
            },
        )
        assert (
            reuse.Version(test_use.test_version)
            == reuse.Version(test_use.__version__)
            == reuse.Version(version)
        )
        assert w[0].category.__name__ == reuse.VersionWarning.__name__


class Restorer:
    def __enter__(self):
        self.locks = set(_shutdown_locks)

    def __exit__(self, arg1, arg2, arg3):
        for lock in set(_shutdown_locks).difference(self.locks):
            lock.release()


@pytest.mark.skipif(is_win, reason="windows reloading")
def test_reloading(reuse):
    fd, file = tempfile.mkstemp(".py", "test_module")
    with Restorer():
        mod = None
        newfile = f"{file}.t"
        for check in range(1):
            with open(newfile, "w") as f:
                f.write(f"def foo(): return {check}")
                f.flush()
            os.rename(newfile, file)
            mod = mod or reuse(Path(file), modes=reuse.reloading)
            while mod.foo() < check:
                pass


def test_suggestion_works(reuse):
    with patch("webbrowser.open"):
        try:
            mod = reuse("example-pypi-package/examplepy", modes=reuse.auto_install)
            assert False, f"Actually returned mod: {mod}"
        except (RuntimeWarning, RuntimeError) as rw:
            last_line = rw.args[0].strip().splitlines()[-1]
            log.info("Using last line as suggested artifact: %s", repr(last_line))
            mod = eval(last_line)
            log.info("suggest artifact returning: %s", mod)
        assert mod


def test_clear_registry(reuse):
    reuse.registry.connection.close()
    try:
        fd, file = tempfile.mkstemp(".db", "test_registry")
        with closing(open(fd, "rb")):
            reuse.registry = reuse._set_up_registry(Path(file))
            reuse.cleanup()
    finally:
        reuse.registry = reuse._set_up_registry()


def installed_or_skip(reuse, name, version=None):
    if not (spec := find_spec(name)):
        pytest.skip(f"{name} not installed")
        return False
    try:
        dist = distribution(spec.name)
    except PackageNotFoundError as pnfe:
        pytest.skip(f"{name} partially installed: {spec=}, {pnfe}")

    if not (
        (ver := dist.metadata["version"]) and (not version or reuse.Version(ver) == reuse.Version(version))
    ):
        pytest.skip(f"found '{name}' v{ver}, but require v{version}")
        return False
    return True


@pytest.mark.parametrize(
    "name, version, hashes",
    (
        (
            "packaging",
            "21.0",
            {"c86254f9220d55e31cc94d69bade760f0847da8000def4dfe1c6b872fd14ff14"},
        ),
        (
            "pytest",
            "6.2.4",
            {"91ef2131a9bd6be8f76f1f08eac5c5317221d6ad1e143ae03894b862e8976890"},
        ),
        # ("pytest-cov", "2.12.1"),
        # ("pytest-env", "0.6.2"),
        # ("requests", "2.24.0"),
        # ("furl", "2.1.2"),
        # ("wheel", "0.36.2"),
        # ("icontract", "2.5.4"),
    ),
)
def test_85(reuse, name, version, hashes):
    if not installed_or_skip(reuse, name, version):
        return
    mod = reuse(name, version=reuse.Version(version))
    assert mod


@pytest.mark.parametrize(
    "name, version, hashes",
    (
        (
            "pytest",
            "6.2.5",
            {"7310f8d27bc79ced999e760ca304d69f6ba6c6649c0b60fb0e04a4a77cacc134"},
        ),
    ),
)
def test_86(reuse, name, version, hashes):
    if not installed_or_skip(reuse, name, version):
        return
    mod = use(
        name,
        version=reuse.Version(version),
        hashes=hashes,
        modes=use.auto_install,
    )
    assert (reuse.Version(modver) if (modver := reuse._get_version(mod=mod)) else version) == reuse.Version(
        version
    )


def test_hash_alphabet():
    H = sha256("hello world".encode("utf-8")).hexdigest()
    assert H == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(H)))


class ScopedArgv(AbstractContextManager):
    def __init__(self, *newargv: list[str]):
        self._oldargv = [*sys.argv]
        self._newargv = newargv

    def __enter__(self, *_):
        sys.argv.clear()
        sys.argv.extend(self._newargv)

    def __exit__(self, *_):
        sys.argv.clear()
        sys.argv.extend(self._oldargv)


@pytest.mark.skipif(is_win, reason="Windows TODO")
def test_setup_py_works(reuse):
    with ScopedArgv("", "bdist_wheel", "-v"):
        result = runpy.run_path("./setup.py")
        assert result


class ScopedCwd(AbstractContextManager):
    def __init__(self, newcwd: Path):
        self._oldcwd = Path.cwd()
        self._newcwd = newcwd

    def __enter__(self, *_):
        os.chdir(self._newcwd)

    def __exit__(self, *_):
        os.chdir(self._oldcwd)


@pytest.mark.skipif(is_win, reason="Windows TODO")
def test_read_wheel_metadata(reuse):
    with ScopedCwd(Path(reuse.main.__file__).parent.parent.parent):
        output = subprocess.check_output([sys.executable, "setup.py", "bdist_wheel", "-v"], shell=False)
        whl_path = Path(
            re.search(r"(?:\')(?:[a-z]+ )*([\w\']+[^\r\n\']*whl)", output.decode(), re.DOTALL).group(1)
        ).absolute()
        assert whl_path.exists()
        assert whl_path.is_file()
        meta = reuse.pimp.archive_meta(whl_path)
        assert meta
        assert meta["name"] == "justuse"
        assert meta["import_relpath"] == "use/__init__.py"


@given(st.text())
def test_jack(inputs):
    assume(inputs.isprintable())
    sha = sha256(inputs.encode("utf-8")).hexdigest()
    assert sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))
