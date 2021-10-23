import threading
from types import ModuleType

from .private import _apply_aspect


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

    def __matmul__(self, other: tuple):
        thing = self.__implementation
        check, pattern, decorator = other
        return _apply_aspect(thing, check, pattern, decorator, aspectize_dunders=False)

    def __call__(self, *args, **kwargs):
        with self.__condition:
            return self.__implementation(*args, **kwargs)
