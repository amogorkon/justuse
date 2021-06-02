"""
A self-documenting, functional way to import modules in Python with advanced features.

Goals:
- version check on the spot, potential version conflicts become obvious (DONE)
- load standalone-modules from online sources with hash-check (DONE)
- auto-reload on file-change (preliminary DONE - works in jupyter)
- pass module-level globals into the importing context (DONE)
- securely auto-install packages (TODO)
- aspect-oriented decorators for everything on import (TODO)
- easy introspection of internal dependency graph (TODO)
- relative imports on online-sources (TBD)

Non-Goal:
Completely replace the import statement.

Note: pathlib.Path and yarl.URL can both be accessed as aliases via use.Path and use.URL

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

__version__ = "0.1.0"

import asyncio
import codecs
import contextlib
import hashlib
import importlib
import inspect
import linecache
import os
import sys
import traceback
from enum import Enum
from functools import singledispatch, update_wrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from warnings import warn

import anyio
import mmh3
import requests
from packaging.version import parse
from yarl import URL


class VersionWarning(Warning):
    pass

class NotReloadableWarning(Warning):
    pass
class NoValidationWarning(Warning):
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

def build_mod(name:str, code:bytes, initial_globals:dict=None, module_path=None) -> ModuleType:
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
    return mod

class SurrogateModule(ModuleType):

    # TODO make this work in general
    @classmethod
    @contextlib.asynccontextmanager
    async def create(cls):
        async with anyio.create_task_group() as tg:
            yield cls(tg)
            tg.cancel_scope.cancel()

    def __init__(self, spec, initial_globals):
        self.__implementation = ModuleType("")
        
        async def __reload():
            last_filehash = None
            while True:
                with open(spec.origin, "rb") as file:
                    current_filehash = hashfileobject(file)
                    if current_filehash != last_filehash:
                        try:
                            file.seek(0)
                            mod = build_mod(spec.name, file.read(), initial_globals)
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
    Path = Path
    URL = URL
    mode = Enum("Mode", "sha256 nodefault")

    def __init__(self):
        self.__using = {}

    @methdispatch
    def __call__(self, thing, *args, **kwargs):
        raise NotImplementedError(f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}.")

    @__call__.register(URL)
    def _use_url(self, 
                url:URL, 
                hash_algo:mode=mode.sha256, 
                hash_value:str=None, 
                initial_globals:dict=None, 
                as_import:str=None,
                default=mode.nodefault,
                ):
        response = requests.get(url)
        if response.status_code != 200:
            raise ModuleNotFoundError(f"Could not load {url} from the interwebs, got a {response.status_code} error.")
        if hash_algo is Use.mode.sha256:
            this_hash = hashlib.sha256(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                if default is not Use.mode.nodefault: 
                    return default
                else:
                    raise UnexpectedHash(f"{this_hash} does not match the expected hash {hash_value} - aborting!")
        else:
            warn(f"""Attempting to import {url} from the interwebs with no validation whatsoever! 
To safely reproduce please use hash_algo="{hash_algo}", hash_value="{this_hash}" """, 
                NoValidationWarning)
        name = url.name
        mod = build_mod(name, response.content, initial_globals)
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
                default=mode.nodefault):
        if path.is_dir():
            if default is not Use.mode.nodefault:
                return default
            else:
                raise RuntimeWarning(f"Can't import directory {path}")
        elif path.is_absolute() and not path.exists():
            if default is not Use.mode.nodefault:
                return default
            else:
                raise ModuleNotFoundError(f"Are you sure '{path.resolve()}' exists?")
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
                mod = SurrogateModule(spec, initial_globals)
                if not getattr(mod, "__reloadable__", False):
                    warn(
                        f"Beware {name} is not flagged as reloadable, things may easily break!",
                        NotReloadableWarning,
                    )
            else:
                # REMINDER: if build_mod is within this block all hell breaks lose!
                with open(path.resolve(), "rb") as file:
                    code = file.read()
                self.__using[f"<{name}>"] = path.resolve()
                mod = build_mod(name, code, initial_globals, path.resolve())
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
                    hash_algo:str=mode.sha256, 
                    hash_value:str=None,
                    default=mode.nodefault,
                    ) -> ModuleType:
        if auto_install:
            if not (version and hash_value):
                raise RuntimeWarning(f"Can't auto-install {name} without a specific version and corresponding hash value")
            with TemporaryDirectory() as directory:
                filename : Path
                with open(filename, "rb") as file:
                    this_hash = securehash_file(file, hash_algo)
                    if this_hash != hash_value:
                        if default is not Use.mode.nodefault:
                            return default
                        else:
                            raise UnexpectedHash(f"Package {name} in temporary {filename} had hash {this_hash}, which does not match the expected {hash_value}, aborting.")

                
            # Nevermind this part, let's make it work for all cases first before optimizing for different cases!
            # # first we need to figure out if we're inside a virtual env
            # def inside_conda():
            #     try:
            #         os.environ["CONDA_DEFAULT_ENV"]
            #         return True
            #     except KeyError:
            #         return False

            # def inside_venv():
            #     return hasattr(sys, 'real_prefix') or sys.base_prefix != sys.prefix

            # if inside_conda() or inside_venv():
            #     subprocess.run(["python", "-m", "pip", "download", f"{name}=={version}"])
            





        spec = importlib.machinery.PathFinder.find_spec(name)
        # builtins may have no spec, let's not mess with those
        if not spec or spec.parent:
            mod = importlib.import_module(name)
        else:
            mod = build_mod(name, spec.loader.get_source(name), initial_globals)
        self.__using[name] = mod, spec, inspect.getframeinfo(inspect.currentframe())

        if version:
            dunder_version = getattr(mod, "__version__", None)
            dunder_version = "" if not dunder_version else parse(str(dunder_version))
            # the whole .version business is ugly :(
            # and then there are LegacyVersions, too.. oh man :|

            try:
                if parse(str(version)) != parse(str(mod.__version__)):
                    warn(
                        f"{name} is expected to be version {version} ,  but got {mod.__version__} instead",
                        VersionWarning,
                    )
            except AttributeError:
                print(f"Cannot determine version for module {name}, continueing.")
        return mod

sys.modules["use"] = Use()
