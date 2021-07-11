
"""
Just use() python code from anywhere - a functional import alternative with advanced features.

Goals/Features:
- inline version checks, user notification on potential version conflicts (DONE)
- securely load standalone-modules from online sources (DONE)
- safely hot auto-reloading of local modules on file changes (DONE)
- pass module-level globals into the importing context (DONE)
- return optional fallback-default object/module if import failed (DONE)
- aspect-oriented decorators for everything callable on import (DONE)
- securely auto-install packages (preliminary DONE, still some kinks with C-extensions)
- support P2P package distribution (TODO)
- unwrap aspect-decorators on demand (TODO)
- easy introspection via internal dependency graph (TODO)
- relative imports on online-sources via URL-aliases (TODO)
- module-level variable placeholders/guards aka "module-properties" (TODO)
- load packages faster while using less memory than classical pip/import - ideal for embedded systems with limited resources (TODO)

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

from __future__ import annotations

import asyncio
import atexit
import codecs
import gzip
import hashlib
import importlib
import inspect
import json
import linecache
import os
import re
import signal
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import zipfile
import zipimport
from collections import defaultdict, namedtuple
from enum import Enum
from functools import singledispatch, update_wrapper, wraps
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
from logging import DEBUG, StreamHandler, getLogger, root
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import mmh3
import requests
import toml
from packaging import tags
from packaging.specifiers import SpecifierSet
from packaging.version import LegacyVersion, Version, parse
from yarl import URL

__version__ = "0.3.2"

_reloaders = {}  # ProxyModule:Reloader
_aspects = {} 
_using = {}

mode = Enum("Mode", "fastfail")
root.addHandler(StreamHandler(sys.stderr))
if "DEBUG" in os.environ: root.setLevel(DEBUG)
log = getLogger(__name__)

# defaults
config = {
        "version_warning": True,
        "debugging": False,
        }

# sometimes all you need is a sledge hammer..
def signal_handler(sig, frame):
    for reloader in _reloaders.values():
        reloader.stop()
    sig, frame
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

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

def hashfileobject(code, sample_threshhold=128 * 1024, sample_size=16 * 1024):
    sample_threshhold, sample_size
    size = len(code)
    hash_tmp = mmh3.hash_bytes(code)
    hash_ = hash_tmp[7::-1] + hash_tmp[16:7:-1]
    enc_size = varint_encode(size)
    return enc_size + hash_[len(enc_size):]

def methdispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

def build_mod(*, name:str, 
                code:bytes, 
                initial_globals:dict, 
                module_path:str, 
                aspectize:dict, 
                default=mode.fastfail) -> ModuleType:
    default
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
    # not catching this causes the most irritating bugs ever!
    try:
        exec(compile(code, f"<{name}>", "exec"), mod.__dict__)
    except: # reraise anything without handling - clean and simple.
        raise
    for (check, pattern), decorator in aspectize.items():
        apply_aspect(mod, check, pattern, decorator)
    return mod

def fail_or_default(default, exception, msg):
    if default is not Use.mode.fastfail:
        return default
    else:
        raise exception(msg)

def apply_aspect(mod:ModuleType, check:callable, pattern:str, decorator:Callable[...]):
    """Apply the aspect as a side-effect, no copy is created."""
    # TODO: recursion?
    parent = mod
    for obj in parent.__dict__.values():
        if check(obj) and re.match(pattern, obj.__qualname__):
            log.debug("Applying aspect to {parent}.{obj.__name__}")
            parent.__dict__[obj.__name__] = decorator(obj)
    return mod

class SurrogateModule(ModuleType):
    def __init__(self, *, name, path, mod, initial_globals, aspectize):
        self.__implementation = mod
        self.__stopped = False

        def __reload_threaded():
            last_filehash = None
            while not self.__stopped:
                with open(path, "rb") as file:
                    code = file.read()
                current_filehash = hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = build_mod(name=name, 
                                        code=code, 
                                        initial_globals=initial_globals,
                                        module_path=path.resolve(),
                                        aspectize=aspectize)
                        self.__implementation = mod
                    except:
                        print(traceback.format_exc())
                last_filehash = current_filehash
                time.sleep(1)

        async def __reload_async():
            last_filehash = None
            while not self.__stopped:
                with open(path, "rb") as file:
                    code = file.read()
                current_filehash = hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = build_mod(name=name, 
                                        code=code, 
                                        initial_globals=initial_globals,
                                        module_path=path.resolve(),
                                        aspectize=aspectize)
                        self.__implementation = mod
                    except:
                        print(traceback.format_exc())
                last_filehash = current_filehash
                await asyncio.sleep(1)
        try:
            # this looks like a hack, but isn't one - 
            # jupyter is running an async loop internally, which works better async than threaded!
            loop = asyncio.get_running_loop()
            loop.create_task(__reload_async())
        except RuntimeError:
            atexit.register(self.__stop)
            self.__thread = threading.Thread(target=__reload_threaded, name=f"reloader__{name}")
            self.__thread.start()

    def __del__(self):
        self.__stopped = True

    def __stop(self):
        self.__stopped = True

    def __getattribute__(self, name):
        if name in ( 
                    "_SurrogateModule__implementation",
                    "_SurrogateModule__stopped",
                    "_SurrogateModule__thread",
                    "_SurrogateModule__stop",
                    ):
            return object.__getattribute__(self, name)
        else:
            return getattr(self.__implementation, name)
    
    def __setattr__(self, name, value):
        if name in (
                    "_SurrogateModule__implementation",
                    "_SurrogateModule__stopped",
                    "_SurrogateModule__thread",
                    "_SurrogateModule__stop",
                    ):
            object.__setattr__(self, name, value)
        else:
            setattr(self.__implementation, name, value)


class ProxyModule(ModuleType):
    def __init__(self, mod):
        self.__implementation = mod
        self.__condition = threading.RLock()

    def __getattribute__(self, name):
        if name in ( 
                    "_ProxyModule__implementation",
                    "_ProxyModule__condition",
                    ""
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

class ModuleReloader:
    def __init__(self, *, proxy, name, path, initial_globals, aspectize):
        self.proxy = proxy
        self.name = name
        self.path = path
        self.initial_globals = initial_globals
        self.aspectize = aspectize
        self._condition = threading.RLock()
        self._stopped = True
        self._thread = None
        
    def start(self):
        assert not (self._thread is not None and not self._thread.is_alive()), "Can't start another reloader thread while one is already running."
        self._stopped = False
        atexit.register(self.stop)
        self._thread = threading.Thread(target=self.run_threaded, name=f"reloader__{self.name}")
        self._thread.start()
    
    def run_threaded(self):
        last_filehash = None
        while not self._stopped:
            with self._condition:
                with open(self.path, "rb") as file:
                    code = file.read()
                current_filehash = hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = build_mod(name=self.name, 
                                        code=code, 
                                        initial_globals=self.initial_globals,
                                        module_path=self.path,
                                        aspectize=self.aspectize)
                        self.proxy._ProxyModule__implementation = mod
                    except:
                        print(traceback.format_exc())
                last_filehash = current_filehash
            time.sleep(1)
    
    def stop(self):
        self._stopped = True
    
    def __del__(self):
        self.stop()
        atexit.unregister(self.stop)


def parse_filename(info:str) -> Optional[dict]:
    """Match the filename and return a dict of parts.
    >>> parse_filename(...)  # TODO add a proper doctest
    {"distribution": .., "version": .., ...} 
    """
    assert isinstance(info, str)
    match:Optional[re.Match] = re.match(
        "(?P<distribution>.*)-"
        "(?P<version>.*)"
        "(?:-(?P<build_tag>.*))?-"
        "(?P<python_tag>.*)-"
        "(?P<abi_tag>.*)-"
        "(?P<platform_tag>.*)\\."
        "(?P<ext>whl|zip|tar|tar\\.gz)",
        info
    )
    return match.groupdict() if match else None

def is_version_satisfied(info:Dict[str,str], sys_version: Version):
    vstr = info if isinstance(info,str) \
                else info["requires_python"] \
                or f'=={info["python_version"]}'
    if not vstr: return False
    vreq = SpecifierSet(vstr)
    return sys_version in vreq

def is_platform_satisfied(info:Dict[str, str], platform_tags:List[str]):
    assert isinstance(info, dict)
    try:
        for it in tags.parse_tag("-".join((info["python_tag"], info["abi_tag"], info["platform_tag"]))):
            if it.platform in platform_tags:
                return True
    except:
        if config["debugging"]: print(traceback.format_exc())
    return False

def is_interpreter_satisfied(info:Dict[str, str], interpreter_tag: str):
    return interpreter_tag in (info["python_tag"], info["abi_tag"])

def find_matching_artifact(
                    urls:List[Dict[str, str]], *, 
                    # for testability
                    sys_version:Version=None,  
                    platform_tags:List[str]=None,
                    interpreter_tag:str=None,
                    ) -> Tuple[str, str]:
    """Pick the best match for our architecture from a list of possible urls and return version and hash."""
    if not sys_version:
        sys_version = Version(".".join(map(str, sys.version_info[0:3])))
    assert isinstance(sys_version, Version)
    if not platform_tags: 
        platform_tags = list(tags._platform_tags())
    assert isinstance(platform_tags, list)
    if not interpreter_tag:
        interpreter_tag = tags.interpreter_name() + tags.interpreter_version()
    assert isinstance(interpreter_tag, str)
    
    return [(info["version"], info["hash"]) for info in urls   # TODO info["hash"] doesn't respect our default hashing algorithm 
                                    if is_version_satisfied(info, sys_version) and 
                                        is_platform_satisfied(info, platform_tags) and 
                                        is_interpreter_satisfied(info, interpreter_tag)][0]
    
def find_latest_working_version(releases: Dict[str, List[Dict[str, str]]], # {version: [{comment_text: str, filename: str, url: str, version: str, hash: str, build_tag: str, python_tag: str, abi_tag: str, platform_tag: str}]}
                                *,
                                #testing
                                sys_version:Version=None,
                                platform_tags:List[str]=None,
                                interpreter_tag:str=None,                                
                                ):
    if not sys_version:
        sys_version = Version(".".join(map(str, sys.version_info[0:3])))
    assert isinstance(sys_version, Version)
    if not platform_tags: 
        platform_tags = list(tags._platform_tags())
    assert isinstance(platform_tags, list)
    if not interpreter_tag:
        interpreter_tag = tags.interpreter_name() + tags.interpreter_version()
    assert isinstance(interpreter_tag, str)
    
    # update the release dicts to hold all info canonically
    for ver, dists in releases.items():
        for d in dists:
            d["version"] = ver # Add version info
            if parsed := parse_filename(d["filename"]):
                d.update(parsed)
    
    for ver, dists in sorted(releases.items(), key=lambda item: item[0]):
        for info in dists:
            if is_version_satisfied(info, sys_version) and \
                is_platform_satisfied(info, platform_tags) and \
                is_interpreter_satisfied(info, interpreter_tag):
                return info["version"], info["hash"]
            
class Use:
    # lift module-level stuff up
    __doc__ = __doc__
    
    # attempt at fix for #23 doesn't work..
    __path__ = str(Path(__file__).resolve().parent)
    
    Path = Path
    URL = URL
    class Hash(Enum):
        sha256 = hashlib.sha256
        
    ModInUse = namedtuple("ModInUse", "name mod path spec frame")

    mode = mode
    config = config
    
    # ALIASES
    isfunction = inspect.isfunction
    ismethod = inspect.ismethod
    isclass = inspect.isclass   
    class VersionWarning(Warning):
        pass
    class NotReloadableWarning(Warning):
        pass
    class NoValidationWarning(Warning):
        pass
    class AmbiguityWarning(Warning):
        pass
    class UnexpectedHash(ImportError):
        pass
    class AutoInstallationError(ImportError):
        pass

    def __init__(self):
        self._using = _using
        self._aspects = _aspects
        self._reloaders = _reloaders

        self.home = Path.home() / ".justuse-python"
        self.home.mkdir(mode=0o755, exist_ok=True)
        (self.home / "packages").mkdir(mode=0o755, exist_ok=True)
        (self.home / "registry.json").touch(mode=0o644, exist_ok=True)
        (self.home / "user_registry.json").touch(mode=0o644, exist_ok=True)
        (self.home / "config.toml").touch(mode=0o644, exist_ok=True)
        (self.home / "config_defaults.toml").touch(mode=0o644, exist_ok=True)
        (self.home / "usage.log").touch(mode=0o644, exist_ok=True)
        # load_registry expects 'self.home' to be set
        self._registry = self.load_registry()
        
        # for the user to copy&paste
        with open(self.home / "default_config.toml", "w") as file:
            toml.dump(config, file)

        with open(self.home / "config.toml") as file:
            config.update(toml.load(file))

        if config["version_warning"]:
            try:
                response = requests.get(f"https://pypi.org/pypi/justuse/json")
                data = response.json()
                max_version = max(parse(version) for version in data["releases"].keys())
                if parse(__version__) < max_version:
                    warn(f"""Justuse is version {parse(__version__)}, but there is a newer version ({max_version}) on PyPI. 
Please consider upgrading via 'python -m pip install -U justuse'""", Use.VersionWarning)
            except:
                warn("Couldn't look up the current version of justuse, you can safely ignore this warning. \n", traceback.format_exc(), "\n \n")

        if config["debugging"]:
            root.setLevel(DEBUG)

    def load_registry(self, registry=None):
        if registry is None:
            registry = getattr(self, "_registry", None)
        registry_version = "0.0.2"
        registry = registry or {}
        try:
            with open(self.home / "registry.json") as file:
                registry.update(json.load(file))
        except json.JSONDecodeError as jde:
            if jde.pos != 0:
                raise
        if "version" in registry \
            and registry["version"] < registry_version:
            print(f"Registry is being upgraded from version "
                    f"{registry.get('version',0)} to version"
                    f"{registry_version}")
            registry["version"] = registry_version
        elif registry and "version" not in registry:
            print(f"Registry is being upgraded from version 0")
            new_registry = {
                "version": registry_version,
                "distributions": (dists := defaultdict(lambda: dict()))
            }
            dists.update(registry)
            registry = new_registry
        if not registry:
            registry.update({
            "version": registry_version,
            "distributions": defaultdict(lambda: dict())
            })
        return registry

    def persist_registry(self):
        with open(self.home / "registry.json", "w") as file:
            json.dump(self._registry, file, indent=2)

    def set_mod(self, *, name, mod, spec, path, frame):
        """Helper to get the order right."""
        self._using[name] = Use.ModInUse(name, mod, path, spec, frame)

    @methdispatch
    def __call__(self, thing, /, *args, **kwargs):
        raise NotImplementedError(f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}.")

    @__call__.register(URL)
    def _use_url(
                self, 
                url:URL, 
                /,*,
                hash_algo:Hash=Hash.sha256, 
                hash_value:str=None, 
                initial_globals:dict=None, 
                as_import:str=None,
                default=mode.fastfail,
                aspectize:dict=None,
                path_to_url:dict=None,
                import_to_use: dict=None,
                ) -> ModuleType:
        exc = None
        path_to_url
        import_to_use
        assert hash_algo in Use.Hash, f"{hash_algo} is not a valid hashing algorithm!"
        
        aspectize = aspectize or {}
        response = requests.get(url)
        if response.status_code != 200:
            raise ImportError(f"Could not load {url} from the interwebs, got a {response.status_code} error.")
        this_hash = hash_algo.value(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                return fail_or_default(default, Use.UnexpectedHash, f"{this_hash} does not match the expected hash {hash_value} - aborting!")
        else:
            warn(f"""Attempting to import from the interwebs with no validation whatsoever! 
To safely reproduce: use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')""", 
                Use.NoValidationWarning)
        name = url.name
        
        try:
            mod = build_mod(name=name, 
                            code=response.content, 
                            module_path=url.path,
                            initial_globals=initial_globals, 
                            aspectize=aspectize)
        except:
            exc = traceback.format_exc()
        if exc:
            return fail_or_default(default, ImportError, exc)
        
        self.set_mod(name=name, mod=mod, spec=None, path=url, frame=inspect.getframeinfo(inspect.currentframe()))
        if as_import:
            assert isinstance(as_import, str), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            assert as_import.isidentifier(), f"as_import must be a valid identifier."
            sys.modules[as_import] = mod
        return mod

    @__call__.register(Path)
    def _use_path(
                self, 
                path:Path, 
                /,*,
                reloading:bool=False,
                initial_globals:dict=None, 
                as_import:str=None,
                default=mode.fastfail,
                aspectize:dict=None,
                path_to_url:dict=None,
                import_to_use: dict=None,
                ) -> ModuleType: 
        aspectize = aspectize or {}
        initial_globals = initial_globals or {}
        exc = None
        mod = None
        path_to_url
        import_to_use

        if path.is_dir():
            return fail_or_default(default, ImportError, f"Can't import directory {path}")
        
        original_cwd = Path.cwd()
        if not path.is_absolute():
            source_dir = getattr(self._using.get(inspect.currentframe().f_back.f_back.f_code.co_filename), "path", None)
            
            # calling from another use()d module
            if source_dir:
                # if calling from an actual file, we take that as starting point
                if source_dir.exists():
                    os.chdir(source_dir.parent)
                    source_dir = source_dir.parent
                else:
                    return fail_or_default(default, NotImplementedError, "Can't determine a relative path from a virtual file.")
            # there are a number of ways to call use() from a non-use() starting point
            else:
                # let's first check if we are running in jupyter
                jupyter = "ipykernel" in sys.modules
                # we're in jupyter, we use the CWD as set in the notebook
                if jupyter:
                    source_dir = original_cwd
                else:
                    # let's see where we started
                    main_mod = __import__("__main__")
                    # if we're calling from a script file e.g. `python3 my/script.py` like pytest unittest
                    if hasattr(main_mod, "__file__"):
                        source_dir = Path(inspect.currentframe().f_back.f_back.f_code.co_filename).resolve().parent
                    else:
                        # interactive startup - use current directory
                        source_dir = original_cwd
        path = source_dir.joinpath(path).resolve()
        if not path.exists():
            os.chdir(original_cwd)
            return fail_or_default(default, ImportError, f"Sure '{path}' exists?")
        os.chdir(path.parent)
        name = path.stem
        if reloading:
            try:
                with open(path, "rb") as file:
                    code = file.read()
                # initial instance, if this doesn't work, just throw the towel
                mod = build_mod(name=name, 
                                code=code, 
                                initial_globals=initial_globals, 
                                module_path=path.resolve(), 
                                aspectize=aspectize
                                )
            except:
                exc = traceback.format_exc()
            if exc:
                return fail_or_default(default, ImportError, exc)
            
            threaded = False
            try:
                # this looks like a hack, but isn't one - 
                # jupyter is running an async loop internally, which works better async than threaded!
                asyncio.get_running_loop()
                
                # Old, working implementation
                mod = SurrogateModule(
                    name=name, 
                    path=path,
                    mod=mod,
                    initial_globals=initial_globals, 
                    aspectize=aspectize
                    )
            # we're dealing with non-async code, we need threading
            # new experimental implementation
            except RuntimeError:
                # can't have the code inside the handler because of "during handling of X, another exception Y happened"
                threaded = True
                
            if threaded:
                mod = ProxyModule(mod)
                reloader = ModuleReloader(
                                        proxy=mod,
                                        name=name, 
                                        path=path, 
                                        initial_globals=initial_globals, 
                                        aspectize=aspectize,
                                        )
                _reloaders[mod] = reloader
                reloader.start()
        
            if not all(inspect.isfunction(value) for key, value in mod.__dict__.items() 
                        if key not in initial_globals.keys() and not key.startswith("__")):
                warn(
                    f"Beware {name} also contains non-function objects, it may not be safe to reload!",
                    Use.NotReloadableWarning,
                )
        else:
            with open(path, "rb") as file:
                code = file.read()
            # the path needs to be set before attempting to load the new module - recursion confusing ftw!
            self.set_mod(name=f"<{name}>", mod=mod, path=path, spec=None, frame=inspect.getframeinfo(inspect.currentframe()))
            try:
                mod = build_mod(name=name, 
                                code=code, 
                                initial_globals=initial_globals, 
                                module_path=path, 
                                aspectize=aspectize)
            except:
                del self._using[f"<{name}>"]
                exc = traceback.format_exc()
        # let's not confuse the user and restore the cwd to the original in any case
        os.chdir(original_cwd)
        if exc:
            return fail_or_default(default, ImportError, exc)
        if as_import:
            assert isinstance(as_import, str), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            sys.modules[as_import] = mod
        self.set_mod(name=f"<{name}>", mod=mod, path=path, spec=None, frame=inspect.getframeinfo(inspect.currentframe()))
        return mod

    @__call__.register(str)
    def _use_str(
                self,
                name:str,
                /,*,
                version:str=None, 
                initial_globals:dict=None, 
                auto_install:bool=False, 
                hash_algo:str=Hash.sha256, 
                hash_value:str=None,
                default=mode.fastfail,
                aspectize=None,
                path_to_url:dict=None,
                import_to_use: dict=None,
                ) -> ModuleType:
        initial_globals = initial_globals or {}
        aspectize = aspectize or {}
        
        assert version is None or isinstance(version, str), "Version must be given as string."
        target_version = parse(version) if version is not None else None
        version = str(target_version) and None  # just validating user input and canonicalizing it
        
        assert not isinstance(target_version, LegacyVersion), "Version must be in a format compatible to https://www.python.org/dev/peps/pep-0440"
        
        exc: str = None
        mod: ModuleType = None
        
        if initial_globals or import_to_use or path_to_url:
            raise NotImplementedError("If you require this functionality, please report it on https://github.com/amogorkon/justuse/issues so we can work out the specifics together.")

        # The "try and guess" behaviour is due to how classical imports work, 
        # which is inherently ambiguous, but can't really be avoided for packages.

        # let's first see if the user might mean something else entirely
        if any(Path(".").glob(f"{name}.py")):
            warn(f"Attempting to load the package '{name}', if you rather want to use the local module: use(use.Path('{name}.py'))", 
                Use.AmbiguityWarning)

        spec = None
        if name in self._using:
            spec = self._using[name].spec
        else:
            if not auto_install:
                spec = importlib.util.find_spec(name)
        
        if spec:
            # let's check if it's a builtin
            builtin = False
            try:
                metadata.PathDistribution.from_name(name)
            except metadata.PackageNotFoundError:  # indeed builtin!
                builtin = True
            if builtin:
                try:
                    mod = spec.loader.create_module(spec)
                    spec.loader.exec_module(mod)  # ! => cache
                    if aspectize:
                        warn("Applying aspects to builtins may lead to unexpected behaviour, but there you go..", RuntimeWarning)
                    for (check, pattern), decorator in aspectize.items():
                        apply_aspect(mod, check, pattern, decorator)
                    self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                    return mod
                except:
                    exc = traceback.format_exc()
                if exc:
                    return fail_or_default(default, ImportError, exc)

            # it seems to be installed in some way, for instance via pip
            if not auto_install:
                try:
                    # feels like cheating, doesn't it
                    mod = importlib.import_module(name)  # ! => cache
                    for (check, pattern), decorator in aspectize.items():
                        apply_aspect(mod, check, pattern, decorator)
                    self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                    warn(f"Classically imported '{name}'. To pin this version use('{name}', version='{metadata.version(name)}')", Use.AmbiguityWarning)
                except:
                    exc = traceback.format_exc()
                if exc:
                    return fail_or_default(default, ImportError, exc)
                
                # we only enforce versions with auto-install
                if target_version:
                    # pure despair :(
                    for check in [
                        "metadata.distribution(name).version",
                        "mod.version",
                        "mod.version()",
                        "mod.__version__"]:
                        try:
                            check_value = eval(check)
                            if isinstance(check_value, str):
                                this_version = parse(check_value)
                                if target_version != this_version:
                                    warn(
                                        f"{name} is expected to be version {target_version} ,  but got {this_version} instead",
                                        Use.VersionWarning,
                                    )
                                    break
                        except:
                            pass
                    else:
                        log.warning(f"Cannot determine version for module {name}, continueing.")
            # spec & auto-install
            else:
                if (metadata.version(name) == target_version) or not(version):
                    if not (version):
                        warn(Use.AmbiguityWarning("No version was provided, even though auto_install was specified! Trying to load classically installed package instead."))
                    try:
                        mod = importlib.import_module(name)  # ! => cache
                        for (check, pattern), decorator in aspectize.items():
                            apply_aspect(mod, check, pattern, decorator)
                        self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                        warn(f"Classically imported '{name}'. To pin this version use('{name}', version='{metadata.version(name)}')", Use.AmbiguityWarning)
                    except:
                        exc = traceback.format_exc()
                    if exc:
                        return fail_or_default(default, ImportError, exc)
                # wrong version => wrong spec
                existing_mod_meta_version = metadata.version(name)
                if existing_mod_meta_version != target_version:
                    log.warning(f"Setting {spec=} to None, because "
                        "the {target_version=} does not match "
                        "the {existing_mod_meta_version=}.")
                    spec = None
        # no spec
        else:
            if not auto_install:
                return fail_or_default(default, ImportError, f"Could not find any installed package '{name}' and auto_install was not requested.")
            
            # the whole auto-install shebang
            package_name, _, module_name = name.partition(".")
            if not module_name:
                module_name = package_name
            
            # PEBKAC
            hit:tuple = None
            
            if target_version and not hash_value:  # let's try to be helpful
                response = requests.get(f"https://pypi.org/pypi/{package_name}/{target_version}/json")
                if response.status_code != 200:
                    raise RuntimeWarning(f"Something bad happened while contacting PyPI for info on {package_name} ( {response.status_code} ), which we tried to look up because you forgot to provide a matching hash_value for the auto-installation.")
                that_hash = None
                try:
                    data = response.json()
                    hit = find_matching_artifact(data["urls"])
                except Exception as _ex:  # ? well.. :)
                    raise
                if hit:
                    version, that_hash = hit
                    raise RuntimeWarning(f"""Failed to auto-install '{package_name}' because hash_value is missing. You may
use("{name}", version="{version}", hash_value="{that_hash}")
""")
                else:
                    raise RuntimeWarning(f"Failed to find any distribution for {package_name} with version {version} that can be run our platform!")
            elif not target_version and hash_value:
                raise RuntimeWarning(f"Failed to auto-install '{package_name}' because version is missing.")
            elif not target_version and not hash_value:
                # let's try to make an educated guess and give a useful suggestion
                response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
                if response.status_code == 404:
                    # possibly typo - PEBKAC
                    raise RuntimeWarning(f"Are you sure package '{package_name}' exists?")
                elif response.status_code != 200:
                    # possibly server problems
                    return fail_or_default(default, Use.AutoInstallationError, f"Tried to look up '{package_name}' but got a {response.status_code} from PyPI.")
                else:
                    try:
                        data = response.json()
                        hit = find_latest_working_version(data["releases"])
                    except KeyError:  # json issues
                        if config["debugging"]:
                            log.error(traceback.format_exc())
                    except:
                        exc = traceback.format_exc()
                    if exc:
                        raise RuntimeWarning("Please specify version and hash for auto-installation. Sadly something went wrong with the JSON PyPI provided, otherwise we could've provided a suggestion.")
                    if hit:
                        version, that_hash = hit
                    else:
                        raise RuntimeWarning(f"We could not find any version or release for {package_name} that could satisfy our requirements!")
                    
                    raise RuntimeWarning(f"""Please specify version and hash for auto-installation of '{package_name}'. 
To get some valuable insight on the health of this package, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version: use("{name}", version="{version}", hash_value="{hash_value}", auto_install=True)
""")

            # all clear, let's check if we pulled it before
            entry = self._registry["distributions"].get(package_name, {}).get(version, {})
            path:Path = None
            if entry and entry["path"]:
                path = Path(entry["path"])
            if entry and path:
                # someone messed with the packages without updating the registry
                if not path.exists():
                    del self._registry["distributions"][package_name][version]
                    self.persist_registry()
                    path = None
                    entry = None
            url:str = None
            that_hash:str = None
            if not path:
                response = requests.get(f"https://pypi.org/pypi/{package_name}/{target_version}/json")
                if response.status_code != 200:
                    return fail_or_default(default, ImportError, f"Tried to auto-install '{package_name}' {target_version} but failed with {response} while trying to pull info from PyPI.")
                try:
                    if not response.json()["urls"]:
                        return fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because no valid URLs to download could be found.")
                    for entry in response.json()["urls"]:
                        url = URL(entry["url"])
                        that_hash = entry["digests"].get(hash_algo.name)
                        if entry["yanked"]:
                            return fail_or_default(default, Use.AutoInstallationError, f"Auto-installation of  '{package_name}' {target_version} failed because the release was yanked from PyPI.")
                        if that_hash == hash_value:
                            break
                    else:
                        return fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because none of the available hashes match the expected hash.")
                except KeyError: # json issues
                    exc = traceback.format_exc()
                if exc:
                    return fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI.")
                # we've got a complete JSON with a matching entry, let's download
                path = self.home / "packages" / Path(url.name).name
                if not path.exists():
                    print("Downloading", url, "...")
                    download_response = requests.get(url, allow_redirects=True)
                    path = self.home / "packages" / Path(url.name).name
                    this_hash:str = hash_algo.value(download_response.content).hexdigest()
                    if this_hash != hash_value:
                        return fail_or_default(default, Use.UnexpectedHash, f"The downloaded content of package {package_name} has a different hash than expected, aborting.")
                    try:
                        with open(path, "wb") as file:
                            file.write(download_response.content)
                        print("Downloaded", path)
                    except:
                        exc = traceback.format_exc()
                    if exc:
                        return fail_or_default(default, Use.AutoInstallationError, exc)
            
            # trying to import directly from zip
            try:
                importer = zipimport.zipimporter(path)
                mod = importer.load_module(module_name)
                print("Direct zipimport of", name, "successful.")
            except:
                if config["debugging"]:
                    log.debug(traceback.format_exc())
                print("Direct zipimport failed, attempting to extract and load manually...")
            
            folder = path.parent / path.stem
            rdists = self._registry["distributions"]
            if not url:
                url = URL(f"file:/{path}")
            
            def create_solib_links(archive: zipfile.ZipFile, folder: Path):
                log.debug(f"create_solib_links({archive=}, {folder=})")
                # EXTENSION_SUFFIXES  == ['.cpython-38-x86_64-linux-gnu.so', '.abi3.so', '.so'] or ['.cp39-win_amd64.pyd', '.pyd']
                entries = archive.getnames() if hasattr(archive, "getnames") \
                    else archive.namelist()
                log.debug(f"archive {entries=}")
                solibs = [*filter(lambda f: any(map(f.endswith, EXTENSION_SUFFIXES)), entries)]
                if not solibs:
                    log.debug(f"No solibs found in archive")
                    return
                # Set up links from 'xyz.cpython-3#-<...>.so' to 'xyz.so'
                log.debug(f"Creating {len(solibs)} symlinks for extensions...")
                log.debug(f"solibs = {solibs}")
                for solib in solibs:
                    sofile = folder / solib
                    log.debug(f"{sofile=}, {folder=}, {solib=}")
                    split_on = [".python", ".cpython", ".cp"]
                    simple_name, os_ext = None, EXTENSION_SUFFIXES[-1]
                    for s in split_on:
                        if not s in sofile.name: continue
                        simple_name = sofile.name.split(s)[0]
                    if simple_name is None: continue
                    link = Path(sofile.parent / f"{simple_name}{os_ext}")
                    if link == sofile: continue
                    log.debug(f"{link=}, {sofile=}")
                    link.unlink(missing_ok=True)
                    link.symlink_to(sofile)
            
            log.debug(f"outside of create_solib_links(...)")
            if package_name not in rdists:
                rdists[package_name] = {}
            if version not in rdists[package_name]:
                rdists[package_name][version] = {}
            # Update package version metadata
            assert url is not None
            rdist_info = rdists[package_name][version]
            rdist_info.update({
                "package": package_name,
                "version": version,
                "url": url.human_repr(),
                "path": str(path) if path else None,
                "folder": folder.absolute().as_uri(),
                "filename": path.name,
                "hash": that_hash
            })
            self.persist_registry()
            
            if not mod:
                if not folder.exists():
                    folder.mkdir(mode=0o755, exist_ok=True)
                    print("Extracting to", folder, "...")
    
                    fileobj = archive = None
                    if path.suffix in (".whl", ".zip"):
                        fileobj = open(tempfile.mkstemp(), "w")
                        archive = zipfile.ZipFile(path, "r")
                    else:
                        fileobj = (gzip.open if path.suffix == ".gz" else open)(path, "r")
                        archive = tarfile.TarFile(fileobj=fileobj, mode="r")
                    with archive as file:
                        with fileobj as _:
                            file.extractall(folder)
                            create_solib_links(file, folder)
                    print("Extracted.")
                original_cwd = Path.cwd()
                
                os.chdir(folder)
                if not sys.path[0] == "":
                    sys.path.insert(0, "")
                importlib.invalidate_caches()
                try:
                  mod = importlib.import_module(module_name)
                  assert parse(mod.__version__) == target_version
                except ImportError:
                  exc = traceback.format_exc()
                log.debug(f"mod = {mod}")
                for key in ("__name__", "__package__", "__path__", "__file__", "__version__", "__author__"):
                    if not hasattr(mod, key): continue
                    rdist_info[key] = getattr(mod, key)
                os.chdir(original_cwd)
            
        self.persist_registry()
        for (check, pattern), decorator in aspectize.items():
            apply_aspect(mod, check, pattern, decorator)
        self.set_mod(name=name, mod=mod, path=None, spec=spec, frame=inspect.getframeinfo(inspect.currentframe()))
        assert mod, "Well shit."
        return mod

sys.modules["use"] = Use()
