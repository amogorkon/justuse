"""Here live all the tests that are expected to fail because their functionality is not implemented yet.
Test-Driven Development is done in the following order:
    1. Create a test that fails.
    2. Write the code that makes the test pass.
    3. Check how long the test took to run.
    4. If it took longer than 1 second, move it to integration tests. Otherwise, move it to unit tests.
"""

import inspect
import logging
import os
from time import time

log = logging.getLogger(".".join((__name__,)))
log.setLevel(logging.DEBUG if "DEBUG" in os.environ else logging.NOTSET)

# ===================


# ===================

for name, func in globals().copy().items():
    if name.startswith("test_"):
        print(f" ↓↓↓↓↓↓↓ {name} ↓↓↓↓↓↓")
        print(inspect.getsource(func))
        now = time()
        func()
        elapsed = time() - now
        print(f"↑↑↑↑↑↑ {name} (in {elapsed:.2f} seconds)")
        print()
