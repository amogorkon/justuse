"""Core tests to quickly check the basic functionality of the library on every edit."""

from justuse import use


def test_use_import_builtin():
    math = use("math")
    assert hasattr(math, "sqrt")
    assert math.sqrt(16) == 4
