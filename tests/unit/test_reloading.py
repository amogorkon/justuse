import sys
import tempfile
from threading import _shutdown_locks

from justuse import Path


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
                import os

                os.rename(newfile, file)
            mod = mod or reuse(Path(file), modes=reuse.reloading)
            while mod.foo() < check:
                pass
