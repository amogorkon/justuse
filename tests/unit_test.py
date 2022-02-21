import os
import re
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable
from contextlib import AbstractContextManager, closing
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from subprocess import STDOUT, check_output
from textwrap import dedent
from threading import _shutdown_locks
from unittest.mock import patch
from warnings import catch_warnings, filterwarnings

import packaging.tags
import packaging.version
import pytest
import requests
from furl import furl as URL
from hypothesis import assume, example, given
from hypothesis import strategies as st
from pytest import fixture, mark, raises, skip

is_win = sys.platform.startswith("win")

__package__ = "tests"
import logging

from use import use
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, is_JACK, num_as_hexdigest
from use.pimp import _parse_name
from use.pypi_model import JustUse_Info, PyPI_Project, PyPI_Release, Version

log = logging.getLogger(".".join((__package__, __name__)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)

use.config["testing"] = True

# this is actually a test!
from tests.simple_funcs import three


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


def test_access_to_home(reuse):
    test = reuse.Path.home() / ".justuse-python/packages/test"
    test.touch(mode=0o644, exist_ok=True)
    with open(test, "w") as file:
        file.write("test")
    assert test.exists()
    test.unlink()
    assert not test.exists()


def test_other_case(reuse):
    with raises(NotImplementedError):
        reuse(2, modes=reuse.fatal_exceptions)


def test_fail_dir(reuse):
    with raises(ImportError):
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
            filterwarnings(action="always", module="use")
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
        filterwarnings(action="always", module="use")
        mod = use("sys")
        assert mod.path is sys.path


def test_classical_install(reuse):
    with warnings.catch_warnings(record=True) as w:
        filterwarnings(action="always", module="use")
        mod = reuse("pytest", version=pytest.__version__, modes=reuse.fatal_exceptions)
        assert mod is pytest or mod._ProxyModule__implementation is pytest
        assert not w


def test_classical_install_no_version(reuse):
    mod = reuse("pytest")
    assert mod is pytest or mod._ProxyModule__implementation is pytest


def test_PEBKAC_hash_no_version(reuse):
    with raises(RuntimeWarning):
        reuse(
            "pytest",
            hashes="asdf",
            modes=reuse.auto_install,
        )


def test_PEBKAC_nonexisting_pkg(reuse):
    # non-existing pkg
    with raises(ImportError):
        reuse(
            "4-^df",
            modes=reuse.auto_install,
            version="0.0.1",
            hashes="asdf",
        )


def test_PEBKAC_impossible_version(reuse):
    # impossible version
    with raises(TypeError):  # version must be either str or tuple
        reuse(
            "pytest",
            modes=reuse.auto_install,
            version=-1,
            hashes="asdf",
        )


def test_autoinstall_PEBKAC(reuse):
    with patch("webbrowser.open"):
        # auto-install requested, but no version or hashes specified
        with raises(RuntimeWarning):
            reuse("pytest", modes=reuse.auto_install)

        # forgot hashes
        with raises(packaging.version.InvalidVersion):
            reuse("pytest", version="-1", modes=reuse.auto_install)


def test_version_warning(reuse):
    # no auto-install requested, wrong version only gives a warning
    with catch_warnings(record=True) as w:
        filterwarnings(action="always", module="use")
        reuse("pytest", version="0.0", modes=reuse.fatal_exceptions)
    assert len(w) != 0
    assert w[0].category is use.VersionWarning


def test_use_global_install(reuse):
    from . import foo

    with raises(NameError):
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
    assert reuse.Version("3.17.3") in reuse._get_package_data_from_pypi("protobuf").releases


def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        filterwarnings(action="always", module="use")
        mod = reuse("furl", version=version)
        assert not w
        assert reuse.Version(mod.__version__) == reuse.Version(version)


def test_classic_import_diff_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with catch_warnings(record=True) as w:
        filterwarnings(action="always", module="use")
        major, minor, patch = version
        mod = reuse(
            "furl",
            version=reuse.Version(major=major, minor=minor, patch=patch + 1),
            modes=reuse.fatal_exceptions,
        )
    assert len(w) != 0
    assert w[0].category == use.VersionWarning


class Restorer:
    def __enter__(self):
        self.locks = set(_shutdown_locks)

    def __exit__(self, arg1, arg2, arg3):
        for lock in set(_shutdown_locks).difference(self.locks):
            lock.release()


def test_reloading(reuse):
    fd, file = tempfile.mkstemp(".py", "test_module")
    with Restorer():
        mod = None
        newfile = f"{file}.t"
        for check in range(1):
            if sys.platform[:3] == "win":
                newfile = file
            with open(newfile, "w") as f:
                f.write(f"def foo(): return {check}")
                f.flush()
            if sys.platform[:3] != "win":
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
        try:
            mod = eval(last_line)
            assert False, f"Actually returned mod: {mod}"
        except (RuntimeWarning, RuntimeError) as rw:
            last_line = rw.args[0].strip().splitlines()[-1]
            log.info("Using last line as suggested artifact: %s", repr(last_line))
        mod = eval(last_line)
        assert mod


def test_clear_registry(reuse):
    reuse.registry.connection.close()
    try:
        fd, file = tempfile.mkstemp(".db", "test_registry")
        with closing(open(fd, "rb")):
            reuse.registry = reuse._set_up_registry(path=Path(file))
            reuse.cleanup()
    finally:
        reuse.registry = reuse._set_up_registry()


def installed_or_skip(reuse, name, version=None):
    if not (spec := find_spec(name)):
        skip(f"{name} not installed")
        return False
    try:
        dist = distribution(spec.name)
    except PackageNotFoundError as pnfe:
        skip(f"{name} partially installed: {spec=}, {pnfe}")

    if not (
        (ver := dist.metadata["version"]) and (not version or reuse.Version(ver) == reuse.Version(version))
    ):
        skip(f"found '{name}' v{ver}, but require v{version}")
        return False
    return True


@mark.parametrize(
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
        (
            "pytest-cov",
            "2.12.1",
            {
                "261ceeb8c227b726249b376b8526b600f38667ee314f910353fa318caa01f4d7",
                "261bb9e47e65bd099c89c3edf92972865210c36813f80ede5277dceb77a4a62a",
            },
        ),
    ),
)
def test_85_pywt_jupyter_ubuntu_case1010(reuse, name, version, hashes):
    """Can't use("pywt", version="1.1.1")
    In jupyter (Lubuntu VM):
    pywt = use("pywt", version="1.1.1")
    """
    if not installed_or_skip(reuse, name, version):
        return
    mod = reuse(name, version=reuse.Version(version))
    assert mod


def test_387_usepath_filename(reuse):
    mod = use(use.Path(".tests/.file_for_test387.py"))
    assert mod


@mark.parametrize(
    "name, version, hashes",
    (
        (
            "pytest",
            "6.2.5",
            {"7310f8d27bc79ced999e760ca304d69f6ba6c6649c0b60fb0e04a4a77cacc134"},
        ),
    ),
)
def test_86_numpy_case1011(reuse, name, version, hashes):
    """Can't use("numpy", version="1.20.0", modes=use.auto_install)

    on windows, py39:
    use("numpy", version="1.20.0", auto_install=True)
    """
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


class ScopedCwd(AbstractContextManager):
    def __init__(self, newcwd: Path):
        self._oldcwd = Path.cwd()
        self._newcwd = newcwd

    def __enter__(self, *_):
        os.chdir(self._newcwd)

    def __exit__(self, *_):
        os.chdir(self._oldcwd)


def test_read_wheel_metadata(reuse):
    bytes = requests.get(
        "https://files.pythonhosted.org/packages/45/80/cdf0df938fe63457f636d859499f4aab3d0411a90fd9472ad720a0b7eab6/justuse-0.5.0.tar.gz"
    ).content
    file = Path(tempfile.mkstemp(".tar.gz", "justuse-0.5.0")[1])
    file.write_bytes(bytes)
    whl_path = file
    if whl_path.exists():
        assert whl_path.exists()
        assert whl_path.is_file()
        meta = reuse.pimp.archive_meta(whl_path)
        assert meta
        assert meta["name"] == "justuse"
        assert meta["import_relpath"].endswith("use/__init__.py")


def test_383_use_name(reuse):
    assert use("pprint").pprint([1, 2, 3]) is None


def test_use_version_upgrade_warning(reuse):
    version = reuse.Version("0.0.0")
    srcdir = reuse.Path(reuse.__spec__.origin).parent.parent
    with ScopedCwd(srcdir):
        output = check_output(
            [
                sys.executable,
                "-c",
                dedent(
                    f"""
                    import os
                    os.environ['USE_VERSION'] = '{version!s}'
                    import use
                    """
                ),
            ],
            encoding="utf-8",
            shell=False,
            stderr=STDOUT,
        )
        match = re.search(r"(?P<category>[a-zA-Z_]+): " r"(?:(?!\d).)* (?P<version>\d+\.\d+\.\d+)", output)
        assert match
        assert match.group("category") == use.VersionWarning.__name__
        assert match.group("version") == str(version)


def test_fraction_of_day(reuse):
    assert reuse.fraction_of_day(datetime(2020, 1, 1)) == 0.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12)) == 500.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 18)) == 750.0
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12, 30, 30)) == 521.180556
    assert reuse.fraction_of_day(datetime(2020, 1, 1, 12, 30, 30, 90000)) == 521.181597


def test_nirvana(reuse):
    with raises(reuse.NirvanaWarning):
        reuse()


@given(st.text())
@example("1t")
def test_jack(text):
    assume(text.isprintable())
    sha = sha256(text.encode("utf-8")).hexdigest()
    assert sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))


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


def test_setup_py_works(reuse):
    with ScopedCwd(Path(__file__).parent.parent):
        result = subprocess.check_output([sys.executable, "setup.py", "--help"], shell=False)
        assert result


def test_443_py_test(reuse):
    try:
        imported = "py" in sys.modules
        import py.test

        if not imported:
            del sys.modules["py"]
    except ImportError:
        skip("py.test is not installed")
        return
    mod = use("py.test")
    assert mod


def test_441_discord(reuse):
    try:
        imported = "discord" in sys.modules
        import discord

        if not imported:
            del sys.modules["discord"]
    except ImportError:
        skip("discord is not installed")
        return
    mod = use("discord")
    assert mod


def test_441_discord(reuse):
    try:
        imported = "discord" in sys.modules
        import discord

        if not imported:
            del sys.modules["discord"]
    except ImportError:
        skip("discord is not installed")
        return
    mod = use("discord")
    assert mod


@mark.parametrize(
    "name,expected",
    [("", (None, None)), ("foo", ("foo", "foo")), ("foo/bar", ("foo", "bar")), ("foo.py", ("foo", "foo.py"))],
)
def test_parse_name(name, expected):
    assert _parse_name(name) == expected
