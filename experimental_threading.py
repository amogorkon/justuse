
import logging
from importlib.util import module_from_spec, spec_from_file_location
from threading import Condition, RLock, Thread
from time import sleep
from types import ModuleType
from typing import Optional
import atexit


logger = logging.getLogger(__name__)


# TODO:
#   Use importlib.util.resolve_name() to resolve relative imports (e.g. `.foo`).
#   <https://docs.python.org/3/library/importlib.html#importlib.util.resolve_name>


class ModuleProxy:
    def __init__(self, module, condition):
        self.__module = module
        self.__condition = condition

    def __getattribute__(self, name: str):
        if name in {'_ModuleProxy__module', '_ModuleProxy__condition'}:
            return object.__getattribute__(self, name)
        logger.debug('Accessing %s', name)
        with self.__condition:
            return getattr(self.__module, name)


class ModuleReloader:
    def __init__(self, spec):
        self._spec = spec
        self._module = None
        self._thread = None
        self._stopped = True
        #self._condition = Condition()
        self._condition = RLock()
        self.module = None

    @classmethod
    def from_file_location(cls, name, location):
        spec = spec_from_file_location(name, location)
        return cls(spec)

    def reload_module(self):
        # TODO: figure out why this doesn't work when running in the background
        # thread, even though I'm outright replacing the module object!
        logger.debug('Reloading %s', module)
        module = module_from_spec(self._spec)
        self._spec.loader.exec_module(module)
        self._module = module
        self.module = ModuleProxy(self._module, self._condition)

    def run(self):
        logger.info('Loop is running.')
        while not self._stopped:
            with self._condition:
                self.reload_module()
                try:
                    logger.debug('%s', self.module.Point2d.x_bounds)
                except AttributeError:
                    logger.debug('MISSING!!')
            sleep(1.0)
        logger.info('Loop is stopped.')

    def stop(self):
        logger.info('Stopping loop.')
        self._stopped = True

    def run_thread(self):
        logger.info('Starting thread.')
        if self._thread is not None and self._thread.is_alive():
            raise ValueError('Thread is already running.')
        self._stopped = False
        self._thread = Thread(target=self.run, name=f'reloader__{self._spec.name}')
        self._thread.start()

    def __del__(self):
        self.stop()
        atexit.unregister(self.stop)

    def __enter__(self):
        self.run_thread()
        return self.module

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def import_reloading(modname, filename):
    return ModuleReloader.from_file_location(modname, filename)


def terminate_thread(thread:Thread):
    """Terminates a python thread from another thread by raising SystemExit in the thread.

    :param thread: a threading.Thread instance
    """
    import ctypes

    if not thread.is_alive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(threadName)s %(levelname)s:%(module)s:%(funcName)s %(message)s')

    reloader = import_reloading('foo', 'foo.py')
    with reloader:
        foo = reloader.module
        print(foo.Point2d.x_bounds)
        sleep(2.0)
        

def reload_stop_all():
    for reloader in _reloaders:
        reloader.stop()

atexit.register(reload_stop_all)