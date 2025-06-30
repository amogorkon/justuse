"""WIP. Don't run via pytest, rather use `py tests/test.py`."""

import inspect

from justuse import Repo, use

# ===========================================


def test_use_test_module():
    # Use the test_module.py from the amogorkon/justuse repo's tests directory
    mod = use(
        Repo.github(
            repo="amogorkon/justuse", path="tests/.tests/test_module.py", ref="main"
        )
    )
    assert mod.test_function() == "Test function executed successfully!"


# ===========================================

for name, func in globals().copy().items():
    if name.startswith("test_"):
        print(f" ↓↓↓↓↓↓↓ {name} ↓↓↓↓↓↓")
        print(inspect.getsource(func))
        func()
        print(f"↑↑↑↑↑↑ {name} ↑↑↑↑↑↑")
        print()
