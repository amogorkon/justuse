from pathlib import Path

from justuse import use


class ScopedCwd:
    def __init__(self, newcwd: Path):
        self._oldcwd = Path.cwd()
        self._newcwd = newcwd

    def __enter__(self, *_):
        import os

        os.chdir(self._newcwd)

    def __exit__(self, *_):
        import os

        os.chdir(self._oldcwd)


def test_387_usepath_filename(reuse):
    with ScopedCwd(Path(__file__).parent):
        mod = use(use.Path(".tests/.file_for_test387.py"))
        assert mod
