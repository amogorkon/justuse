import io
from contextlib import redirect_stdout
from unittest.mock import patch


def test_454_bad_metadata(reuse):
    name = "pyinputplus"
    with patch("webbrowser.open"), io.StringIO() as buf, redirect_stdout(buf):
        try:
            reuse(name, modes=reuse.auto_install)
        except RuntimeWarning:
            version = buf.getvalue().splitlines()[-1].strip()
        try:
            reuse(name, version=version, modes=reuse.auto_install)
        except RuntimeWarning:
            recommended_hash = buf.getvalue().splitlines()[-1].strip()
        mod = reuse(
            name,
            version=version,
            hashes={recommended_hash},
            modes=reuse.auto_install | reuse.no_cleanup,
        )
        assert mod
