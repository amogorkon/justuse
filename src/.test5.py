import use

mod = use(
    "sqlalchemy",
    version="0.7.1",
    hashes={"5ef95d19c31a8cd3905c697be0a7e94e70ab1926ecd4159c3e6c1cf01fc3c492"},
    # SQLAlchemy-0.7.1.tar.gz (2.3 MB) - only a single artifact
    # Uploaded Jun 5, 2011 source
    # but it's bugged on import - it's looking for time.clock(), which isn't a thing (anymore)
    modes=use.auto_install,
)
