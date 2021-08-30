import builtins
import importlib
import inspect
from types import GetSetDescriptorType, MethodType

imports = {}

import inspect
import sys

sys_paths = {}


class SysPath(list):
    def append(self, item):
        origin = inspect.currentframe().f_back.f_code.co_filename
        print(origin, type(origin), sys_paths[origin])
        if origin in sys_paths:
            print("appending to other syspath", self, item)
            sys_paths[origin].append(item)
        else:
            print("appending to original", item)
            super().append(item)

    def insert(self, item, position):
        origin = inspect.currentframe().f_back.f_code.co_filename
        print(origin)
        if origin in sys_paths:
            print("inserting to other syspath", item, position)
            sys_paths[origin].insert(item, position)
        else:
            print("inserting to original", item, position)
            super().insert(item, position)


sys.path = SysPath(sys.path)

sys_paths[r"F:\Dropbox (Privat)\code\justuse\tests\.tests\.test2.py"] = []


def use(
    name,
    *args,
    import_to_use=None,
):

    print("using:", name)
    return importlib.import_module(name)


def decorator(func):
    def wrapper(name, *args):
        to_use: str
        if name in to_use:
            return use(name, *args)
        imports[name] = inspect.currentframe().f_back.f_code.co_filename
        return func(name, *args)

    return wrapper


builtins.__import__ = decorator(__import__)
use("numpy", import_to_use=["re"])
