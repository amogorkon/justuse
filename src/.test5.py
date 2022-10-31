import use

mod = use(
    "sqlalchemy",
    version="0.7.1",
    hashes={"45df54adf"},  # the real thing takes 12 sec to run, way too long for a test
    # SQLAlchemy-0.7.1.tar.gz (2.3 MB) - only a single artifact
    # Uploaded Jun 5, 2011 source
    # but it's bugged on import - it's looking for time.clock(), which isn't a thing (anymore)
    modes=use.auto_install,
    default=None,
)
assert mod is None
