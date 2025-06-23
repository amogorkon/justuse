import os
import warnings
from warnings import filterwarnings

from pytest import raises

from justuse import URL, Hash, Path, use


def test_access_to_home(reuse):
    test = reuse.config.packages / "test"
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
    foo_path = Path(__file__).parent.parent / ".tests" / "foo.py"
    print(f"loading foo module via Path('{foo_path}'))")
    mod = reuse(Path(foo_path), initial_globals={"a": 42})
    assert mod.test() == 42


def test_internet_url(reuse):
    foo_uri = "https://raw.githubusercontent.com/greyblue9/justuse/3f783e6781d810780a4bbd2a76efdee938dde704/tests/foo.py"
    print(f"loading foo module via URL({foo_uri}))")
    mod = reuse(
        URL(foo_uri),
        initial_globals={"a": 42},
        hash_algo=Hash.sha256,
        hash_value="b136efa1d0dab3caaeb68bc41258525533d9058aa925d3c0c5e98ca61200674d",
    )
    assert mod.test() == 42


def test_module_package_ambiguity(reuse):
    original_cwd = os.getcwd()
    try:
        os.chdir(Path(__file__).parent.parent / ".tests")
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
    import sys

    with warnings.catch_warnings(record=True) as w:
        filterwarnings(action="always", module="use")
        mod = use("sys")
        assert mod.path is sys.path
