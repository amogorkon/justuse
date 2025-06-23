import pytest


def test_use_global_install(reuse):
    from . import foo

    with pytest.raises(NameError):
        foo.bar()
    reuse.install()
    assert foo.bar()
    reuse.uninstall()
    del foo
