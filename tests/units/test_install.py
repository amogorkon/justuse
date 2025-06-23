import warnings

import pytest


def test_classical_install(reuse):
    import pytest

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(action="always", module="use")
        mod = reuse("pytest", version=pytest.__version__, modes=reuse.fatal_exceptions)
        assert mod is pytest or mod._ProxyModule__implementation is pytest
        assert not w


def test_classical_install_no_version(reuse):
    import pytest

    mod = reuse("pytest")
    assert mod is pytest or mod._ProxyModule__implementation is pytest


def test_PEBKAC_hash_no_version(reuse):
    with pytest.raises(RuntimeWarning):
        reuse(
            "pytest",
            hashes="addf",
            modes=reuse.auto_install,
        )


def test_PEBKAC_nonexisting_pkg(reuse):
    with pytest.raises(ImportError):
        reuse(
            "4-^df",
            modes=reuse.auto_install,
            version="0.0.1",
            hashes="addf",
        )


def test_PEBKAC_impossible_version(reuse):
    with pytest.raises(TypeError):
        reuse(
            "pytest",
            modes=reuse.auto_install,
            version=-1,
            hashes="asdf",
        )


def test_autoinstall_PEBKAC(reuse):
    from unittest.mock import patch

    import packaging.version

    with patch("webbrowser.open"):
        with pytest.raises(RuntimeWarning):
            reuse("pytest", modes=reuse.auto_install)
        with pytest.raises(packaging.version.InvalidVersion):
            reuse("pytest", version="-1", modes=reuse.auto_install)


def test_version_warning(reuse):
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(action="always", module="use")
        reuse("pytest", version="0.0", modes=reuse.fatal_exceptions)
    assert len(w) != 0
    import justuse as use_mod

    assert w[0].category is use_mod.VersionWarning
