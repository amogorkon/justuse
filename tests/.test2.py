from types import ModuleType
import asyncio
import contextlib

class SurrogateModule(ModuleType):
    def __init__(self):
        self.__implementation = ModuleType("")

    def __getattribute__(self, name):
        if name in ("_SurrogateModule__name", "_SurrogateModule__reloading", "_SurrogateModule__implementation"):
            return object.__getattribute__(self, name)
        else:
            return getattr(self.__implementation, name)
    
    def __setattr__(self, name, value):
        if name == "_SurrogateModule__implementation":
            object.__setattr__(self, name, value)
        setattr(self.__implementation, name, value)
            
mod = SurrogateModule()