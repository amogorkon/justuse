from types import ModuleType
import threading
from importlib import import_module

class ProxyModule(ModuleType):
    def __init__(self, mod):
        self.__implementation = mod
        self.__condition = threading.RLock()
    
    @property
    def path(self):
        print("accessing sys.path..")
        return 5

    def __getattribute__(self, name):
        if name in ("_ProxyModule__implementation", "_ProxyModule__condition", "", "path"):
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

import sys
sys.modules["sys"] = ProxyModule(sys)
sys.x = 4

import asdf