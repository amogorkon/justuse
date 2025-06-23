import warnings

from justuse import use


def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(action="always", module="use")
        mod = reuse("furl", version=version)
        assert not w
        assert reuse.Version(mod.__version__) == reuse.Version(version)


def test_classic_import_diff_version(reuse):
    version = reuse.Version(__import__("furl").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(action="always", module="use")
        major, minor, patch = version
        mod = reuse(
            "furl",
            version=reuse.Version(major=major, minor=minor, patch=patch + 1),
            modes=reuse.fatal_exceptions,
        )
    assert len(w) != 0
    assert w[0].category == use.VersionWarning
