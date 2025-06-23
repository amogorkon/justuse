import pytest

from justuse import auto_install, no_cleanup, use
from justuse.main import Modes, config


class ReuseProxy:
    def __call__(self, *args, **kwargs):
        return use(*args, **kwargs)

    @property
    def config(self):
        return config

    @property
    def auto_install(self):
        return auto_install

    @property
    def no_cleanup(self):
        return no_cleanup

    @property
    def fatal_exceptions(self):
        return Modes.fatal_exceptions

    @property
    def reloading(self):
        return Modes.reloading

    @property
    def verbose(self):
        return Modes.verbose

    @property
    def silent(self):
        return Modes.silent

    @property
    def dryrun(self):
        return Modes.dryrun

    @property
    def offline(self):
        return Modes.offline

    @property
    def no_cleanup_mode(self):
        return Modes.no_cleanup

    @property
    def auto_install_mode(self):
        return Modes.auto_install


@pytest.fixture
def reuse():
    return ReuseProxy()
