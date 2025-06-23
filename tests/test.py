"""WIP. Don't run via pytest, rather use `py tests/test.py`."""

import inspect

from justuse import use

# ===========================================


def test_use():
    use("time")


# ===========================================

for name, func in globals().copy().items():
    if name.startswith("test_"):
        print(f" ↓↓↓↓↓↓↓ {name} ↓↓↓↓↓↓")
        print(inspect.getsource(func))
        func()
        print(f"↑↑↑↑↑↑ {name} ↑↑↑↑↑↑")
        print()
