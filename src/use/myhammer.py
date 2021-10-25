import atexit
import sys
import threading


# sometimes all you need is a sledge hammer..
def releaser(cls):
    old_locks = [*threading._shutdown_locks]
    new_locks = threading._shutdown_locks
    reloaders = sys.modules["use"]._reloaders
    releaser = cls()

    def release():
        return releaser(locks=set(new_locks).difference(old_locks), reloaders=reloaders)

    atexit.register(release)
    return cls


@releaser
class ShutdownLockReleaser:
    def __call__(cls, *, locks: list, reloaders: list):
        for lock in locks:
            lock.unlock()
        for reloader in reloaders:
            if hasattr(reloader, "stop"):
                reloader.stop()
        for lock in locks:
            lock.unlock()
