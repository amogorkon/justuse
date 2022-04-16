import ast
import contextlib
import sys
from importlib import metadata
from _frozen_importlib import BuiltinImporter
from pathlib import Path

from pprint import pprint

import beartype
import numpy

def is_builtin(name, mod):
    if name in sys.builtin_module_names:
        return True
    
    if hasattr(mod, "__file__"):
        relpath = Path(mod.__file__).parent.relative_to(
                (Path(sys.executable).parent / "lib"))
        if relpath == Path():
            return True
        if relpath.parts[0] == "site-packages":
            return False
    return True

    

def get_imports_from_module(mod):
    if not hasattr(mod, "__file__"):
        return
    with open(mod.__file__, "rb") as file:
        with contextlib.suppress(ValueError):
            for x in ast.walk(ast.parse(file.read())):
                if isinstance(x, ast.Import):
                    name = x.names[0].name
                    if (mod := sys.modules.get(name)) and not is_builtin(name, mod):
                        yield name
                if isinstance(x, ast.ImportFrom):
                    name = x.module
                    if (mod := sys.modules.get(name)) and not is_builtin(name, mod):
                        yield name


def get_submodules(mod, visited=None, results=None):
    if results is None:
        results = set()
    if visited is None:
        visited = set()
    for name in get_imports_from_module(mod):
        if name in visited:
            continue
        visited.add(name)
        results.add(name)
        for x in get_imports_from_module(sys.modules[name]):
            results.add(x)
            get_submodules(sys.modules[x], visited, results)
    return results


pprint(list(get_submodules(beartype)))
print(len(get_submodules(beartype)))
