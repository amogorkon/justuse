from justuse import auto_install, no_cleanup, use


def test_51_sqlalchemy_failure_default_to_none(reuse):
    mod = use(
        "sqlalchemy",
        version="0.7.1",
        hashes={"45df54adf"},
        modes=auto_install | no_cleanup,
        default=None,
    )
    assert mod is None
