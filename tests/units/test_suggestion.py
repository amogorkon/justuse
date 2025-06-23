import io
from contextlib import redirect_stdout
from unittest.mock import patch


def test_suggestion_works(reuse):
    name = "package-example"
    with patch("webbrowser.open"), io.StringIO() as buf, redirect_stdout(buf):
        try:
            mod = reuse(name, modes=reuse.auto_install)
            assert False, f"Actually returned mod: {mod}"
        except RuntimeWarning:
            version = buf.getvalue().splitlines()[-1].strip()
        try:
            mod = reuse(name, version=version, modes=reuse.auto_install)
            assert False, f"Actually returned mod: {mod}"
        except RuntimeWarning:
            recommended_hash = buf.getvalue().splitlines()[-1].strip()
        mod = reuse(
            name, version=version, hashes={recommended_hash}, modes=reuse.auto_install
        )
        assert mod
