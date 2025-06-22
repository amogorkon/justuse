import asyncio
import atexit
import hashlib
import threading
import time
import traceback
from types import ModuleType

from icontract import require

from .modutils import _build_mod, _modules_are_compatible


class ProxyModule(ModuleType):
    def __init__(self, mod):
        self.__implementation = mod
        self.__condition = threading.RLock()

    def __getattribute__(self, name):
        if name in (
            "_ProxyModule__implementation",
            "_ProxyModule__condition",
            "",
            "__class__",
            "__metaclass__",
            "__instancecheck__",
        ):
            return object.__getattribute__(self, name)
        with self.__condition:
            return getattr(self.__implementation, name)

    def __setattr__(self, name, value):
        if name in (
            "_ProxyModule__implementation",
            "_ProxyModule__condition",
        ):
            object.__setattr__(self, name, value)
            return
        with self.__condition:
            setattr(self.__implementation, name, value)

    def __rmatmul__(self, *args, **kwargs):
        return ProxyModule.__matmul__(self, *args, **kwargs)

    # forwarding method calls - allowing even weirder modules?
    # https://github.com/GrahamDumpleton/wrapt/blob/develop/src/wrapt/wrappers.py

    def __call__(*args, **kwargs):
        def _unpack_self(self, *args):
            return self, args

        self, args = _unpack_self(*args)

        return self._ProxyModule__implementation(*args, **kwargs)


class ModuleReloader:
    def __init__(self, *, proxy, name, path, pkg_name, initial_globals):
        self.proxy = proxy
        "ProxyModule that we refer to."
        self.name = name
        self.path = path
        self.pkg_name = pkg_name
        self.initial_globals = initial_globals
        self._condition = threading.RLock()
        self._stopped = True
        self._thread = None

    def start_async(self):
        loop = asyncio.get_running_loop()
        loop.create_task(self.run_async())

    @require(lambda self: self._thread is None or self._thread.is_alive())
    def start_threaded(self):
        self._stopped = False
        atexit.register(self.stop)
        self._thread = threading.Thread(
            target=self.run_threaded, name=f"reloader__{self.name}"
        )
        self._thread.start()

    async def run_async(self):
        last_filehash = None
        while not self._stopped:
            with open(self.path, "rb") as file:
                code = file.read()
            current_filehash = hashlib.blake2b(code).hexdigest()
            if current_filehash != last_filehash:
                try:
                    mod = _build_mod(
                        mod_name=self.name,
                        code=code,
                        initial_globals=self.initial_globals,
                        module_path=self.path.resolve(),
                    )
                    if not _modules_are_compatible(self.proxy, mod):
                        continue
                    self.proxy.__implementation = mod
                except KeyError:
                    traceback.print_exc()
            last_filehash = current_filehash
            await asyncio.sleep(1)

    def run_threaded(self):
        last_filehash = None
        while not self._stopped:
            with self._condition:
                with open(self.path, "rb") as file:
                    code = file.read()
                current_filehash = hashlib.blake2b(code).hexdigest()
                if current_filehash != last_filehash:
                    try:
                        mod = _build_mod(
                            mod_name=self.name,
                            code=code,
                            initial_globals=self.initial_globals,
                            module_path=self.path,
                        )
                        if not _modules_are_compatible(self.proxy, mod):
                            continue
                        self.proxy._ProxyModule__implementation = mod
                    except KeyError:
                        traceback.print_exc()
                last_filehash = current_filehash
            time.sleep(1)

    def stop(self):
        self._stopped = True

    def __del__(self):
        self.stop()
