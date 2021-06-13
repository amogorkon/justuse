"""
A self-documenting, explicit, functional way to import modules in Python with advanced features.

Goals/Features:
- version check on the spot, potential version conflicts become obvious (DONE)
- securely load standalone-modules from online sources (DONE)
- safely auto-reloading of local modules on edit (preliminary DONE - works in jupyter)
- pass module-level globals into the importing context (DONE)
- return optional fallback-default object/module if import failed (DONE)
- aspect-oriented decorators for everything callable on import (DONE)
- securely auto-install packages (TODO)
- support P2P package distribution (TODO)
- unwrap aspect-decorators on demand (TODO)
- easy introspection via internal dependency graph (TODO)
- relative imports on online-sources via URL-aliases (TODO)
- module-level variable placeholders/guards aka "module-properties" (TODO)

Non-Goal:
Completely replace the import statement.

Notes: 
pathlib.Path and yarl.URL can both be accessed as aliases via use.Path and use.URL
inspect.isfunction, .ismethod and .isclass also can be accessed via their aliases use.isfunction, use.ismethod and use.isclass

Examples:
>>> import use

# equivalent to `import numpy as np` with explicit version check
>>> np = use("numpy", version="1.1.1")
>>> np.version == "1.1.1"
True

# equivalent to `from pprint import pprint; pprint(dictionary)` but without assigning 
# pprint to a global variable, thus no namespace pollution
>>> use("pprint").pprint([1,2,3])
[1,2,3]
# equivalent to sys.path manipulation, then `import tools` with a reload(tools) every second
>>> tools = use(use.Path("/media/sf_Dropbox/code/tools.py"), reloading=True)

# it is possible to import standalone modules from online sources
# with immediate sha1-hash-verificiation before execution of the code like
>>> utils = use(use.URL("https://raw.githubusercontent.com/PIA-Group/BioSPPy/7696d682dc3aafc898cd9161f946ea87db4fed7f/biosppy/utils.py"),
                    hash_value="95f98f25ef8cfa0102642ea5babbe6dde3e3a19d411db9164af53a9b4cdcccd8")

# to auto-install a certain version (within a virtual env and pip in secure hash-check mode) of a package you can do
>>> np = use("numpy", version="1.1.1", auto_install=True, hash_value=["9879de676"])

File-Hashing inspired by 
- https://github.com/kalafut/py-imohash
- https://github.com/fmoo/python-varint/blob/master/varint.py

:author: use-github@anselm.kiefner.de (Anselm Kiefner)
:license: MIT
"""

import asyncio
import codecs
import contextlib
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import linecache
import os
import re
import sys
import traceback

from enum import Enum
from enum import Flag
from functools import singledispatch
from functools import update_wrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from warnings import warn

import anyio
import mmh3
import requests

from packaging.version import parse
from yarl import URL

__version__ = "0.2.2"


class VersionWarning(Warning):
    pass

class NotReloadableWarning(Warning):
    pass

class NoValidationWarning(Warning):
    pass

class AmbiguityWarning(Warning):
    pass

class ModuleNotFoundError(ImportError):
    pass

class UnexpectedHash(ImportError):
    pass


def varint_encode(number):
    """Pack `number` into varint bytes"""
    buf = b''
    while True:
        towrite = number & 0x7f
        number >>= 7
        if number:
            buf += bytes((towrite | 0x80,))
        else:
            buf += bytes((towrite,))
            break
    return buf

def hashfileobject(file, sample_threshhold=128 * 1024, sample_size=16 * 1024):
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0, os.SEEK_SET)
    if size < sample_threshhold or sample_size < 1:
        data = file.read()
    else:
        data = file.read(sample_size)
        file.seek(size//2)
        data += file.read(sample_size)
        file.seek(-sample_size, os.SEEK_END)
        data += file.read(sample_size)

    hash_tmp = mmh3.hash_bytes(data)
    hash_ = hash_tmp[7::-1] + hash_tmp[16:7:-1]
    enc_size = varint_encode(size)
    return enc_size + hash_[len(enc_size):]

def securehash_file(file, hash_algo):
    BUF_SIZE = 65536
    if hash_algo is Use.mode.sha256:
        file_hash = hashlib.sha256()
    while True:
        data = file.read(BUF_SIZE)
        if not data:
            break
        file_hash.update(data)
    return file_hash.hexdigest()

def methdispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

def build_mod(name:str, code:bytes, initial_globals:dict=None, module_path=None, aspectize:dict=None) -> ModuleType:
    mod = ModuleType(name)
    mod.__dict__.update(initial_globals or {})
    mod.__file__ = module_path

    code_text = codecs.decode(code)
    # module file "<", ">" chars are specially handled by inspect
    linecache.cache[f"<{name}>"] = (
    len(code), # size of source code
    None, # last modified time; None means there is no physical file
    [*map( # a list of lines, including trailing newline on each
        lambda ln: ln+"\x0a",
        code_text.splitlines())
    ],
    mod.__file__, # file name, e.g. "<mymodule>" or the actual path to the file
    )
    exec(compile(code, f"<{name}>", "exec"), mod.__dict__)
    for (check, pattern), decorator in aspectize:
        apply_aspect(mod, check, pattern, decorator)
    return mod

def fail_or_default(default, exception, msg):
    if default is not Use.mode.fastfail:
        return default
    else:
        raise exception(msg)

def apply_aspect(mod:ModuleType, check:callable, pattern:str, decorator:callable):
    # TODO: recursion?
    parent = mod
    for name, obj in parent.__dict__.items():
        if check(obj) and re.match(pattern, obj.__qualname__):
            # TODO: logging?
            parent.__dict__[obj.__name__] = decorator(obj)

class SurrogateModule(ModuleType):

    # TODO make this work in general
    @classmethod
    @contextlib.asynccontextmanager
    async def create(cls):
        async with anyio.create_task_group() as tg:
            yield cls(tg)
            tg.cancel_scope.cancel()

    def __init__(self, spec, initial_globals, aspectize):
        self.__implementation = ModuleType("")
        
        async def __reload():
            last_filehash = None
            while True:
                with open(spec.origin, "rb") as file:
                    current_filehash = hashfileobject(file)
                    if current_filehash != last_filehash:
                        try:
                            file.seek(0)
                            mod = build_mod(spec.name, file.read(), initial_globals, aspectize=aspectize)
                            # TODO: check for different AST or globals
                            self.__implementation = mod
                        except Exception as e:
                            print(e, traceback.format_exc())
                    last_filehash = current_filehash
                await asyncio.sleep(1)
        self.__reloading = asyncio.get_event_loop().create_task(__reload())  # TODO: this isn't ideal

    def __del__(self):
        self.__reloading.cancel()

    def __getattribute__(self, name):
        if name in ("_SurrogateModule__reloading",
                    "_SurrogateModule__implementation",
                    ):
            return object.__getattribute__(self, name)
        else:
            return getattr(self.__implementation, name)
    
    def __setattr__(self, name, value):
        if name in ("_SurrogateModule__reloading",
                    "_SurrogateModule__implementation",
                    ):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__implementation, name, value)

class Use:
    __doc__ = __doc__  # module's __doc__ above
    __version__ = __version__  # otherwise setup.py can't find it
    Path = Path
    URL = URL
    class Hash(Enum):
        sha256 = hashlib.sha256

    mode = Enum("Mode", "fastfail")
    
    isfunction = inspect.isfunction
    ismethod = inspect.ismethod
    isclass = inspect.isclass

    def __init__(self):
        self.__using = {}
        self.__aspectized = {}

    @methdispatch
    def __call__(self, thing, *args, **kwargs):
        raise NotImplementedError(f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}.")

    @__call__.register(URL)
    def _use_url(self, 
                url:URL, 
                hash_algo:Hash=Hash.sha256, 
                hash_value:str=None, 
                initial_globals:dict=None, 
                as_import:str=None,
                default=mode.fastfail,
                aspectize=None,
                ):
        assert hash_algo in Use.Hash, f"{hash_algo} is not a valid hashing algorithm!"
        
        aspectize = aspectize or {}
        response = requests.get(url)
        if response.status_code != 200:
            raise ModuleNotFoundError(f"Could not load {url} from the interwebs, got a {response.status_code} error.")
        this_hash = hash_algo.value(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                return fail_or_default(default, UnexpectedHash, f"{this_hash} does not match the expected hash {hash_value} - aborting!")
        else:
            warn(f"""Attempting to import {url} from the interwebs with no validation whatsoever! 
To safely reproduce please use hash_algo="{hash_algo}", hash_value="{this_hash}" """, 
                NoValidationWarning)
        name = url.name
        mod = build_mod(name, response.content, initial_globals, aspectize=aspectize)
        self.__using[name] = mod, inspect.getframeinfo(inspect.currentframe())
        if as_import:
            assert isinstance(as_import, str), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            sys.modules[as_import] = mod

        return mod

    @__call__.register(Path)
    def _use_path(self, 
                path:Path, 
                reloading:bool=False,
                initial_globals:dict=None, 
                as_import=None,
                default=mode.fastfail,
                aspectize=None):  # sourcery skip: remove-redundant-pass
        aspectize = aspectize or {}
        if path.is_dir():
            return fail_or_default(default, ImportError, f"Can't import directory {path}")
        elif path.is_absolute() and not path.exists():
            return fail_or_default(default, ModuleNotFoundError, f"Are you sure '{path.resolve()}' exists?")
        else:
            original_cwd = os.getcwd()
            if not path.is_absolute():
                source_dir = self.__using.get(inspect.currentframe().f_back.f_back.f_code.co_filename)
                # if calling from an actual file, we take that as starting point
                if source_dir is not None and source_dir.exists():
                    os.chdir(source_dir.parent)
                    path = path.resolve()
                # calling from jupyter for instance, we use the cwd set there, no guessing
                else:
                    pass
            name = path.stem
            # TODO: replacing previously loaded module
            pass
            if reloading:
                spec = importlib.machinery.PathFinder.find_spec(name)
                mod = SurrogateModule(spec, initial_globals, aspectize)
                if not all(inspect.isfunction(obj) for obj in mod.__dict__.values() 
                                                        if obj not in initial_globals.values()):
                    warn(
                        f"Beware {name} also contains non-function objects, it may not be safe to reload!",
                        NotReloadableWarning,
                    )
            else:
                # REMINDER: if build_mod is within this block all hell breaks lose!
                with open(path.resolve(), "rb") as file:
                    code = file.read()
                self.__using[f"<{name}>"] = path.resolve()
                mod = build_mod(name, code, initial_globals, path.resolve(), aspectize=aspectize)
            # let's not confuse the user and restore the cwd to the original in any case
            os.chdir(original_cwd)
            if as_import:
                assert isinstance(as_import, str), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
                sys.modules[as_import] = mod
            
            return mod

    @__call__.register(str)
    def _use_str(self, name:str, 
                    version:str=None, 
                    initial_globals:dict=None, 
                    auto_install:bool=False, 
                    hash_algo:str=Hash.sha256, 
                    hash_value:str=None,
                    default=mode.fastfail,
                    aspectize=None,
                    ) -> ModuleType:

        aspectize = aspectize or {}
        # let's first check if it's installed already somehow
        spec = importlib.machinery.PathFinder.find_spec(name)

        if any(Path(".").glob(f"{name}.py")):
            warn(f"Attempting to load the package '{name}', if you rather want to use the local module: use(use.Path('{name}.py'))", 
                AmbiguityWarning)

        # couldn't find any installed package
        if not spec:
            if not auto_install:
                return fail_or_default(default, ImportError, f"{name} is not installed and auto-install was not requested.")

            # TODO: raise appropriate detailed warnings and give helpful info from the json to fix the issue
            if not (version and hash_value):
                raise RuntimeWarning(f"Can't auto-install {name} without a specific version and corresponding hash value")

            response = requests.get(f"https://pypi.org/pypi/{name}/{version}/json")
            if response != 200:
                return fail_or_default(default, ImportError, f"Tried to auto-install {name} {version} but failed with {response} while trying to pull info from PyPI.")
            try:
                if not response.json()["urls"]:
                    return fail_or_default(default, ImportError, f"Tried to auto-install {name} {version} but failed because no valid URLs to download could be found.")
                for entry in response.json()["urls"]:
                    url = entry["url"]
                    that_hash = entry["digests"].get(hash_algo.name)
                    filename = entry["filename"]
                    # special treatment?
                    yanked = entry["yanked"]
                    if that_hash == hash_value:
                        break
                else:
                    return fail_or_default(default, ImportError, f"Tried to auto-install {name} {version} but failed because none of the available hashes match the expected hash.")
            except KeyError:
                return fail_or_default(default, ImportError, f"Tried to auto-install {name} {version} but failed because there was a problem with the JSON from PyPI.")

            with TemporaryDirectory() as directory:
                # TODO: chdir etc
                # download the file
                with open(filename, "wb") as file:
                    pass
                # check the hash
                this_hash = securehash_file(file, hash_algo)
                if this_hash != hash_value:
                    return fail_or_default(default, UnexpectedHash, f"Package {name} in temporary {filename} had hash {this_hash}, which does not match the expected {hash_value}, aborting.")
                # load it
            # now that we got something, we can load it
            spec = importlib.machinery.PathFinder.find_spec(name)

        # now there should be a valid spec defined

        # builtins may have no spec, let's not mess with those
        if not spec or spec.parent:
            mod = importlib.import_module(name)
            # not using build_mod, so we need to do this from here
            for (check, pattern), decorator in aspectize.items():
                apply_aspect(mod, check, pattern, decorator)
        else:
            mod = build_mod(name, spec.loader.get_source(name), initial_globals, aspectize=aspectize)
        self.__using[name] = mod, spec, inspect.getframeinfo(inspect.currentframe())

        if version:
            try:
                # packages usually keep their metadata in seperate files, not in some __version__ variable
                if parse(str(version)) != parse(metadata.version(name)):
                    warn(
                        f"{name} is expected to be version {version} ,  but got {mod.__version__} instead",
                        VersionWarning,
                    )
            except AttributeError:
                print(f"Cannot determine version for module {name}, continueing.")

        return mod

sys.modules["use"] = Use()
