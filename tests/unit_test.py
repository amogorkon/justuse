
import functools
import os
import re
import sys
import tempfile
import warnings
from contextlib import closing
from importlib.util import find_spec
from importlib.metadata import distribution, PackageNotFoundError
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


def suggested_artifact(name, *args, **kwargs):
    try:
        mod = use(name, *args, modes=1, **kwargs)
        assert False, f"Actually returned mod: {mod}"
    except (RuntimeWarning, RuntimeError) as rw:
        assert rw.args
        for message in rw.args:
            return _parse_warning_message("%s" % message)
    assert False, "Did not find a suggested artifact"


def _parse_warning_message(message: str):
    RW_REGEX = re.compile(
        r"version=((?:(?!, \w+=).)+).*hashes=((?:(?!, \w+=).)+)", re.DOTALL
    )
    version_evalstr, hashes_evalstr = RW_REGEX.findall(message)[0]
    log.debug(
        "eval'ing the following string from rw message [%s]:\n "
        " hashes_evalstr: [%s], version_evalstr: [%s]",
        message,
        hashes_evalstr,
        version_evalstr,
    )
    hashes = eval(hashes_evalstr)
    version = eval(version_evalstr)
    result = (version, hashes)
    log.debug("eval'ed to the following: %s", result)
    assert isinstance(
        hashes, (set, str)
    ), f"Wrong object type is given in the warning: {message!r}"
    return result


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
                mod = reuse(URL(foo_uri), initial_globals={"a": 42})
                assert mod.test() == 42
    finally:
        os.chdir(orig_cwd)


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
    mod = reuse("pytest", modes=reuse.fatal_exceptions)
    assert mod is pytest or mod._ProxyModule__implementation is pytest


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
    with pytest.raises(TypeError):  # version must be either str or tuple
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
    assert False == reuse._is_version_satisfied(info.get("requires_python", ""), sys_version)

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
    assert "3.17.3" in data["releases"]


def test_parse_filename(reuse):
    assert reuse._parse_filename("protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl") == {
        "distribution": "protobuf",
        "version": "1.19.5",
        "python_tag": "cp36",
        "abi_tag": "cp36m",
        "platform_tag": "macosx_10_9_x86_64",
        "ext": "whl",
    }


def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("furl", version=version)
        assert not w
        assert reuse.Version(mod.__version__) == version


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
        assert test_use.test_version == test_use.__version__ == version
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
    sugg = suggested_artifact("example-pypi-package/examplepy")
    assert sugg
    mod = reuse(
        "example-pypi-package/examplepy",
        version=sugg[0],
        hashes=sugg[1],
        modes=use.auto_install,
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
@pytest.mark.parametrize("name, version", (("numpy", "1.19.3"),))


def test_86_numpy(reuse, name, version):
    use = reuse  # for the eval() later
    with pytest.raises(RuntimeWarning) as w:
        reuse(name, version=version, modes=reuse.auto_install)
    assert w
    recommendation = str(w.value).split("\n")[-1].strip()
    mod = eval(recommendation)
    assert mod.__name__ == reuse._parse_name(name)[1]
    assert mod.__version__ == version
    return mod  # for the redownload test


def test_clear_registry(reuse):
    reuse.registry.connection.close()
    try:
        fd, file = tempfile.mkstemp(".db", "test_registry")
        with closing(open(fd, "rb")):
            reuse.registry = reuse._set_up_registry(Path(file))
            reuse.cleanup()
    finally:
        reuse.registry = reuse._set_up_registry()


def installed_or_skip(name, version=None):
    if not (spec := find_spec(name)):
        pytest.skip(f"{name} not installed")
        return False
    try:
        dist = distribution(spec.name)
    except PackageNotFoundError as pnfe:
        pytest.skip(f"{name} partially installed: {spec=}, {pnfe}")
    
    if not ((ver := dist.metadata["version"])
       and (version is None or ver == version)):
        pytest.skip(f"found '{name}' v{ver}, but require v{version}")
        return False
    return True
@pytest.mark.parametrize(
    "name, version, hashes",
    (
        ('packaging', '21.0', {'c86254f9220d55e31cc94d69bade760f0847da8000def4dfe1c6b872fd14ff14'}),
        ('pytest', '6.2.4', {'91ef2131a9bd6be8f76f1f08eac5c5317221d6ad1e143ae03894b862e8976890'}),
        # ("pytest-cov", "2.12.1"),
        # ("pytest-env", "0.6.2"),
        # ("requests", "2.24.0"),
        # ("furl", "2.1.2"),
        # ("wheel", "0.36.2"),
        # ("icontract", "2.5.4"),
    )\
)


def test_85(reuse, name, version, hashes):
    if not installed_or_skip(name, version):
        return
    mod = reuse(name, version=reuse.Version(version))
    assert mod
@pytest.mark.parametrize(
    "name, version, hashes",
    (
        ('pytest', '6.2.5', {'7310f8d27bc79ced999e760ca304d69f6ba6c6649c0b60fb0e04a4a77cacc134'}),
        ('PyYAML', '5.4.1', {'d483ad4e639292c90170eb6f7783ad19490e7a8defb3e46f97dfe4bacae89122'}),
        ('pyzmq', '22.2.1', {'b4428302c389fffc0c9c07a78cad5376636b9d096f332acfe66b321ae9ff2c63'}),
        ('scipy', '1.7.1', {'611f9cb459d0707dd8e4de0c96f86e93f61aac7475fcb225e9ec71fecdc5cebf'}),
        ('setuptools', '57.4.0', {'a49230977aa6cfb9d933614d2f7b79036e9945c4cdd7583163f4e920b83418d6'}),
        ('setuptools-scm', '6.0.1', {'c3bd5f701c8def44a5c0bfe8d407bef3f80342217ef3492b951f3777bd2d915c'}),
        ('terminado', '0.12.1', {'09fdde344324a1c9c6e610ee4ca165c4bb7f5bbf982fceeeb38998a988ef8452'}),
        ('testpath', '0.5.0', {'8044f9a0bab6567fc644a3593164e872543bb44225b0e24846e2c89237937589'}),
        ('tornado', '6.1', {'a48900ecea1cbb71b8c71c620dee15b62f85f7c14189bdeee54966fbd9a0c5bd'}),
        ('traitlets', '5.1.0', {'03f172516916220b58c9f19d7f854734136dd9528103d04e9bf139a92c9f54c4'}),
        ('typing-extensions', '3.10.0.2', {'f1d25edafde516b146ecd0613dabcc61409817af4766fbbcfb8d1ad4ec441a34'}),
        # ("pytest-cov", "2.12.1"),
        # ("pytest-env", "0.6.2"),
        # ("requests", "2.24.0"),
        # ("furl", "2.1.2"),
        # ("wheel", "0.36.2"),
        # ("icontract", "2.5.4"),
    )
)


def test_86(reuse, name, version, hashes):
    if not installed_or_skip(name, version):
        return
    mod = use(
        name,
        version=reuse.Version(version),
        hashes=hashes,
        modes=use.auto_install,
    )
    assert reuse._get_version(mod=mod) == reuse.Version(version)

