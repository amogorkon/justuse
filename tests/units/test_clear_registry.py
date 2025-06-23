import tempfile
from contextlib import closing
from pathlib import Path


def test_clear_registry(reuse):
    reuse.registry.connection.close()
    try:
        fd, file = tempfile.mkstemp(".db", "test_registry")
        with closing(open(fd, "rb")):
            reuse.registry = reuse._set_up_registry(path=Path(file))
            reuse.cleanup()
    finally:
        reuse.registry = reuse._set_up_registry()
