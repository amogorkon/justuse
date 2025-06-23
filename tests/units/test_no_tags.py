import io
from contextlib import redirect_stdout
from unittest.mock import patch


def test_454_no_tags(reuse):
    name, version = "pyspark", "3.1.2"
    with patch("webbrowser.open"), io.StringIO() as buf, redirect_stdout(buf):
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
