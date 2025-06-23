import sys

from pytest import skip


def test_443_py_test(reuse):
    try:
        imported = "py" in sys.modules
        import py.test

        if not imported:
            del sys.modules["py"]
    except ImportError:
        skip("py.test is not installed")
        return
    mod = reuse("py.test")
    assert mod
