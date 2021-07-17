


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
from copy import copy
from enum import Enum
from functools import singledispatch, update_wrapper, wraps
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
from itertools import chain
from logging import DEBUG, StreamHandler, getLogger, root
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

try:
    from pip._internal.utils.compatibility_tags import get_supported
except ImportError:
    get_supported = lambda: []

import mmh3
import packaging
import requests
import toml
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from yarl import URL

__version__ = "0.4.0"

_reloaders = {}  # ProxyModule:Reloader
_aspects = {} 
_using = {}

class Version(Version):  # Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves. This is necessary to compare sys.version_info with Version.
    def __iter__(self):
        yield from self.release

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

def methdispatch(func): # singledispatch for methods
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper

class SurrogateModule(ModuleType):
    def __init__(self, *, name, path, mod, initial_globals, aspectize):
        self.__implementation = mod
        self.__stopped = False

        def __reload_threaded():
            last_filehash = None
            while not self.__stopped:
                with open(path, "rb") as file:
                    code = file.read()
                current_filehash = Use._hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = Use._build_mod(name=name, 
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
                current_filehash = Use._hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = Use._build_mod(name=name, 
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
                current_filehash = Use._hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = Use._build_mod(name=self.name, 
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

class Use:
    # lift module-level stuff up
    __doc__ = __doc__
    __version__ = __version__
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
    class MissingHash(ValueError):
        pass

    def __init__(self):        
        self._using = _using
        self._aspects = _aspects
        self._reloaders = _reloaders

        self.home = Path.home() / ".justuse-python"
        try:
            self.home.mkdir(mode=0o755, exist_ok=True)
        except PermissionError:
            # this should fix the permission issues on android #80
            self.home = Path(tempfile.mkdtemp(prefix="justuse_"))
        (self.home / "packages").mkdir(mode=0o755, exist_ok=True)
        (self.home / "registry.json").touch(mode=0o644, exist_ok=True)
        (self.home / "user_registry.json").touch(mode=0o644, exist_ok=True)
        (self.home / "config.toml").touch(mode=0o644, exist_ok=True)
        (self.home / "config_defaults.toml").touch(mode=0o644, exist_ok=True)
        (self.home / "usage.log").touch(mode=0o644, exist_ok=True)
        self._registry:dict = Use._load_registry(self.home / "registry.json")
        self._user_registry:dict = Use._load_registry(self.home / "user_registry.json")
        Use._merge_registry(self._registry, self._user_registry)
        
        # for the user to copy&paste
        with open(self.home / "default_config.toml", "w") as file:
            toml.dump(config, file)

        with open(self.home / "config.toml") as file:
            config.update(toml.load(file))

        if config["version_warning"]:
            try:
                response = requests.get(f"https://pypi.org/pypi/justuse/json")
                data = response.json()
                max_version = max(Version(version) for version in data["releases"].keys())
                if Version(__version__) < max_version:
                    warn(f"""Justuse is version {Version(__version__)}, but there is a newer version ({max_version}) on PyPI. 
Please consider upgrading via 'python -m pip install -U justuse'""", Use.VersionWarning)
            except:
                log.debug(traceback.format_exc())  # we really don't need to bug the user about this (either pypi is down or internet is broken)
                
        if config["debugging"]:
            root.setLevel(DEBUG)

    def install(self):
        # yeah, really.. __builtins__ sometimes appears as a dict and other times as a module, don't ask me why
        if isinstance(__builtins__, dict):
            __builtins__["use"] = self
        elif isinstance(__builtins__, ModuleType):
            setattr(__builtins__, "use", self)
        else:
            raise RuntimeWarning("__builtins__ is something unexpected")
    
    def uninstall(self):
        if isinstance(__builtins__, dict):
            if "use" in __builtins__:
                del __builtins__["use"]
        elif isinstance(__builtins__, ModuleType):
            if hasattr(__builtins__, "use"):
                delattr(__builtins__, "use")
        else:
            RuntimeWarning("__builtins__ is something unexpected")

    def persist_registry(self):
        assert all(version is not None for version in (dist.keys() for dist in self._registry["distributions"].values()))

        with open(self.home / "registry.json", "w") as file:
            text = "### WARNING: This file is automatically generated. Any manual changes will be overwritten. For persistent alterations, please use the user_registry.json ###\n" + json.dumps(self._registry, indent=2)
            file.write(text)

    def cleanup(self):
        """Bring registry and downloaded packages in sync.
        
        First all packages are removed that don't have a matching registry entry, then all registry entries that don't have a matching package.
        """
        def delete_folder(path) :
            for sub in path.iterdir() :
                if sub.is_dir() :
                    delete_folder(sub)
                else :
                    sub.unlink()
            path.rmdir()
        # let's first take care of unregistered package folders
        for package_path in (self.home/"packages").iterdir():
            if package_path.stem not in (Path(version["path"]).stem 
                                         for dist in  self._registry["distributions"].values()
                                         for version in dist.values()
                                         ):
                if package_path.is_dir():
                    delete_folder(package_path)
                else:
                    package_path.unlink()
        
        # let's clean up the registry entries that don't have a matching package
        
        # can't iterate over a changing dict :|
        for name in list(self._registry["distributions"].keys()):
            for version in list(self._registry["distributions"][name].keys()):
                if not Path(self._registry["distributions"][name][version]["path"]).exists():
                    del self._registry["distributions"][name][version]
        
        for name in list(self._registry["distributions"].keys()):
            if self._registry["distributions"][name] == {}:
                del self._registry["distributions"][name]
        self.persist_registry()

    def set_mod(self, *, name, mod, spec, path, frame):
        """Helper to get the order right."""
        self._using[name] = Use.ModInUse(name, mod, path, spec, frame)
        
    # hoisted functions - formerly module globals
    # staticmethods because module globals aren't reachable in tests while there should be no temptation for side effects via self

    @staticmethod
    def _get_platform_tags() -> set:
        return set(get_supported()) | set(packaging.tags._platform_tags())

    @staticmethod
    def _parse_filename(filename:str) -> dict:
        """Match the filename and return a dict of parts.
        >>> parse_filename("numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl")
        {'distribution': 'numpy', 'version': '1.19.5', 'build_tag': None, 'python_tag': 'cp36', 'abi_tag': 'cp36m', 'platform_tag': 'macosx_10_9_x86_64', 'ext': 'whl'}
        """
        assert isinstance(filename, str)
        match:Optional[re.Match] = re.match(
            "(?P<distribution>.*)-"
            "(?P<version>.*)"
            "(?:-(?P<build_tag>.*))?-"
            "(?P<python_tag>.*)-"
            "(?P<abi_tag>.*)-"
            "(?P<platform_tag>.*)\\."
            "(?P<ext>whl|zip|tar|tar\\.gz)",
            filename
        )
        if not match:
            log.debug(f"filename {filename} could not be matched")
        return match.groupdict() if match else {}

    @staticmethod
    def _is_version_satisfied(info:Dict[str,str], sys_version: Version):        
        # https://warehouse.readthedocs.io/api-reference/json.html
        # https://packaging.pypa.io/en/latest/specifiers.html
        specifier = info.get("requires_python", "")  # SpecifierSet("") matches anything, no need to artificially lock down versions at this point
        # f.. you PyPI
        if specifier is None:
            specifier = ""
        return sys_version in SpecifierSet(specifier)

    @staticmethod
    def _is_platform_compatible(info:Dict[str, str], platform_tags:set, include_sdist=False):
        assert isinstance(info, dict) and isinstance(platform_tags, set)
        info.update(Use._parse_filename(info["filename"]))  # filename as API, seriously WTF...
        # source is compatible with any platform by default, just need to check the version
        if info["python_version"] == "source" and include_sdist:
            return True
        our_python_tag = "".join((
                                packaging.tags.interpreter_name(),
                                packaging.tags.interpreter_version()))
        python_tag = info.get("python_tag", "")
        platform_tag = info.get("platform_tag", "")
        for one_platform_tag in platform_tag.split("."):
            is_match = one_platform_tag in platform_tags and \
                         our_python_tag == python_tag
            log.debug("%s: \"%s\" in platform_tags and %s == %s", is_match, one_platform_tag, python_tag, our_python_tag)
            if is_match:
                return True
        return False

    @staticmethod
    def _find_matching_artifact(
                        urls:List[Dict[str, str]], 
                        hash_algo:str=Hash.sha256.name,
                        *, 
                        # for testability
                        sys_version:Version=None,  
                        platform_tags:List[str]=None,
                        interpreter_tag:str=None,
                        ) -> Tuple[str, str]:
        """Pick from a list of possible urls and return the hash of the best fitting artifact."""
        if not sys_version:
            sys_version = Version(".".join(map(str, sys.version_info[0:3])))
        assert isinstance(sys_version, Version)
        if not platform_tags: 
            platform_tags = Use._get_platform_tags()
        assert isinstance(platform_tags, set)
        if not interpreter_tag:
            interpreter_tag = packaging.tags.interpreter_name() + packaging.tags.interpreter_version()
        assert isinstance(interpreter_tag, str)
        
        results = [info["digests"][hash_algo] for info in 
                    sorted(urls, 
                            key=lambda info: info.get("packagetype", ""))  # pre-sorting by type should ensure that we prefer binary packages over raw source
                        if Use._is_version_satisfied(info, sys_version) and
                            Use._is_platform_compatible(info, platform_tags)
                    ]
        if results:
            return results[0]
        else: 
            return None

    @staticmethod
    def _find_latest_working_version(releases: Dict[str, List[Dict[str, str]]], # {version: [{comment_text: str, filename: str, url: str, version: str, hash: str, build_tag: str, python_tag: str, abi_tag: str, platform_tag: str}]}
                                    *,
                                    hash_algo:str,
                                    #testing
                                    sys_version:Version=None,
                                    platform_tags:List[str]=None,
                                    interpreter_tag:str=None,                                
                                    ):
        assert isinstance(releases, dict)
        assert isinstance(hash_algo, str)
        if not sys_version:
            sys_version = Version(".".join(map(str, sys.version_info[0:3])))
        assert isinstance(sys_version, Version)
        if not platform_tags: 
            platform_tags = list(packaging.tags._platform_tags())
        assert isinstance(platform_tags, list)
        platform_tags = set(platform_tags)
        if not interpreter_tag:
            interpreter_tag = packaging.tags.interpreter_name() + packaging.tags.interpreter_version()
        assert isinstance(interpreter_tag, str)
        
        
        # update the release dicts to hold all info canonically
        # be aware, the json returned from pypi ["releases"] can contain empty lists as dists :/
        for ver, dists in releases.items():
            if not dists:
                continue
            for d in dists:
                d["version"] = ver # Add version info
                d.update(Use._parse_filename(d["filename"]))
        
        for ver, dists in sorted(releases.items(), key=lambda item: Version(item[0]), reverse=True):
            print(ver)
            if not dists:
                continue
            for info in dists:
                if Use._is_version_satisfied(info, sys_version) and \
                    Use._is_platform_compatible(info, platform_tags):
                    hash_value = info["digests"].get(hash_algo)
                    if not hash_value:
                        raise Use.MissingHash(f"No hash digest found in "
                            "release distribution for {hash_algo=}")
                    return info["version"], hash_value
    
    @staticmethod
    def _load_registry(path):
        registry_version = "0.0.2"
        registry = {}
        with open(path) as file:
            # json doesn't have comments, so we need to manually skip the first line warning for the user
            lines = file.readlines()
            if not lines:  # might be an empty file
                registry.update({
                "version": registry_version,
                "distributions": defaultdict(lambda: dict())
                })
                return registry
            registry.update(json.loads("\n".join(filter(lambda s: not s.startswith("#"), lines))))  # Now comments in user_registry.json are ignored, too

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
        return registry
    
    @staticmethod
    def _merge_registry(target, source):
        for k, v in source.items():
            if k in target:
                if isinstance(target[k], list):
                    target[k] += v
                elif isinstance(target[k], dict):
                    for k2, v2 in v.items():
                        target[k][k2] = v2
                else:
                    target[k] = v

    @staticmethod
    def _varint_encode(number):
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

    @staticmethod
    def _hashfileobject(code, sample_threshhold=128 * 1024, sample_size=16 * 1024):
        sample_threshhold, sample_size
        size = len(code)
        hash_tmp = mmh3.hash_bytes(code)
        hash_ = hash_tmp[7::-1] + hash_tmp[16:7:-1]
        enc_size = Use._varint_encode(size)
        return enc_size + hash_[len(enc_size):]

    @staticmethod
    def _build_mod(*, name:str, 
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
            Use._apply_aspect(mod, check, pattern, decorator)
        return mod

    @staticmethod
    def _fail_or_default(default, exception, msg):
        if default is not Use.mode.fastfail:
            return default
        else:
            raise exception(msg)

    @staticmethod
    def _apply_aspect(mod:ModuleType, check:callable, pattern:str, decorator:Callable[...]):
        """Apply the aspect as a side-effect, no copy is created."""
        # TODO: recursion?
        parent = mod
        for obj in parent.__dict__.values():
            if check(obj) and re.match(pattern, obj.__qualname__):
                log.debug("Applying aspect to {parent}.{obj.__name__}")
                parent.__dict__[obj.__name__] = decorator(obj)
        return mod

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
                return Use._fail_or_default(default, Use.UnexpectedHash, f"{this_hash} does not match the expected hash {hash_value} - aborting!")
        else:
            warn(f"""Attempting to import from the interwebs with no validation whatsoever! 
To safely reproduce: use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')""", 
                Use.NoValidationWarning)
        name = url.name
        
        try:
            mod = Use._build_mod(name=name, 
                            code=response.content, 
                            module_path=url.path,
                            initial_globals=initial_globals, 
                            aspectize=aspectize)
        except:
            exc = traceback.format_exc()
        if exc:
            return Use._fail_or_default(default, ImportError, exc)
        
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
            return Use._fail_or_default(default, ImportError, f"Can't import directory {path}")
        
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
                    return Use._fail_or_default(default, NotImplementedError, "Can't determine a relative path from a virtual file.")
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
            return Use._fail_or_default(default, ImportError, f"Sure '{path}' exists?")
        os.chdir(path.parent)
        name = path.stem
        if reloading:
            try:
                with open(path, "rb") as file:
                    code = file.read()
                # initial instance, if this doesn't work, just throw the towel
                mod = Use._build_mod(name=name, 
                                code=code, 
                                initial_globals=initial_globals, 
                                module_path=path.resolve(), 
                                aspectize=aspectize
                                )
            except:
                exc = traceback.format_exc()
            if exc:
                return Use._fail_or_default(default, ImportError, exc)
            
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
                mod = Use._build_mod(name=name, 
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
            return Use._fail_or_default(default, ImportError, exc)
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
                hash_algo:Hash=Hash.sha256, 
                hash_value:str=None,
                default=mode.fastfail, 
                aspectize=None,
                path_to_url:dict=None,
                import_to_use: dict=None,
                fatal_exceptions:bool=False,
                exchange_sys_path:bool=False
                ) -> ModuleType:
        initial_globals = initial_globals or {}
        aspectize = aspectize or {}
        
        assert version is None or isinstance(version, str), "Version must be given as string."
        target_version:Version = Version(version) if version else None
        # just validating user input and canonicalizing it
        version:str = str(target_version) if target_version else None
        assert version if target_version else version is target_version, \
          "Version must be None if target_version is None; otherwise, they must both have a value."
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
                        Use._apply_aspect(mod, check, pattern, decorator)
                    self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                    return mod
                except:
                    exc = traceback.format_exc()
                if exc:
                    return Use._fail_or_default(default, ImportError, exc)

            # it seems to be installed in some way, for instance via pip
            if not auto_install:
                try:
                    # feels like cheating, doesn't it
                    mod = importlib.import_module(name)  # ! => cache
                    for (check, pattern), decorator in aspectize.items():
                        Use._apply_aspect(mod, check, pattern, decorator)
                    self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                    warn(f"Classically imported '{name}'. To pin this version use('{name}', version='{metadata.version(name)}')", Use.AmbiguityWarning)
                except:
                    if fatal_exceptions: raise
                    exc = traceback.format_exc()
                if exc:
                    return Use._fail_or_default(default, ImportError, exc)
                
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
                                this_version = Version(check_value)
                                if target_version != this_version:
                                    warn(
                                        f"{name} is expected to be version {target_version} ,  but got {this_version} instead",
                                        Use.VersionWarning,
                                    )
                                    break
                        except:
                            if fatal_exceptions: raise
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
                            Use._apply_aspect(mod, check, pattern, decorator)
                        self.set_mod(name=name, mod=mod, spec=spec, path=None, frame=inspect.getframeinfo(inspect.currentframe()))
                        warn(f"Classically imported '{name}'. To pin this version use('{name}', version='{metadata.version(name)}')", Use.AmbiguityWarning)
                    except:
                        if fatal_exceptions: raise
                        exc = traceback.format_exc()
                    if exc:
                        return Use._fail_or_default(default, ImportError, exc)
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
                return Use._fail_or_default(default, ImportError, f"Could not find any installed package '{name}' and auto_install was not requested.")
            
            # the whole auto-install shebang
            package_name, _, module_name = name.partition(".")
            if not module_name:
                module_name = package_name
            
            # PEBKAC
            hit:tuple = None
            
            if target_version and not hash_value:  # let's try to be helpful
                response = requests.get(f"https://pypi.org/pypi/{package_name}/{target_version}/json")
                if response.status_code == 404:
                    raise RuntimeWarning(f"Are you sure {package_name} with version {version} exists?")
                elif response.status_code != 200:
                    raise RuntimeWarning(f"Something bad happened while contacting PyPI for info on {package_name} ( {response.status_code} ), which we tried to look up because you forgot to provide a matching hash_value for the auto-installation.")
                that_hash = None
                data = response.json()
                that_hash = Use._find_matching_artifact(data["urls"])
                if that_hash:
                    raise RuntimeWarning(f"""Failed to auto-install '{package_name}' because hash_value is missing. This may work:
use("{name}", version="{version}", hash_value="{that_hash}", auto_install=True)
""")
                else:
                    raise RuntimeWarning(f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!")
            elif not target_version and hash_value:
                raise RuntimeWarning(f"Failed to auto-install '{package_name}' because no version was specified.")
            elif not target_version and not hash_value:
                # let's try to make an educated guess and give a useful suggestion
                response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
                if response.status_code == 404:
                    # possibly typo - PEBKAC
                    raise RuntimeWarning(f"Are you sure package '{package_name}' exists? Could not find any package with that name on PyPI.")
                elif response.status_code != 200:
                    # possibly server problems
                    return Use._fail_or_default(default, Use.AutoInstallationError, f"Tried to look up '{package_name}' but got a {response.status_code} from PyPI.")
                else:
                    try:
                        data = response.json()
                        hit = Use._find_latest_working_version(data["releases"], hash_algo=hash_algo.name)
                        if hit:
                          version, hash_value = hit
                    except KeyError:  # json issues
                        if fatal_exceptions: raise
                        if config["debugging"]:
                            log.error(traceback.format_exc())
                    except:
                        if fatal_exceptions: raise
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
                version = str(target_version)
                if response.status_code != 200:
                    return Use._fail_or_default(default, ImportError, f"Tried to auto-install '{package_name}' {target_version} but failed with {response} while trying to pull info from PyPI.")
                try:
                    if not response.json()["urls"]:
                        return Use._fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because no valid URLs to download could be found.")
                    for entry in response.json()["urls"]:
                        url = URL(entry["url"])
                        that_hash = entry["digests"].get(hash_algo.name)
                        if entry["yanked"]:
                            return Use._fail_or_default(default, Use.AutoInstallationError, f"Auto-installation of  '{package_name}' {target_version} failed because the release was yanked from PyPI.")
                        if that_hash == hash_value:
                            break
                    else:
                        return Use._fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because none of the available hashes match the expected hash.")
                except KeyError: # json issues
                    if fatal_exceptions: raise
                    exc = traceback.format_exc()
                if exc:
                    return Use._fail_or_default(default, Use.AutoInstallationError, f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI.")
                # we've got a complete JSON with a matching entry, let's download
                path = self.home / "packages" / Path(url.name).name
                if not path.exists():
                    print("Downloading", url, "...")
                    download_response = requests.get(url, allow_redirects=True)
                    path = self.home / "packages" / Path(url.name).name
                    this_hash:str = hash_algo.value(download_response.content).hexdigest()
                    if this_hash != hash_value:
                        return Use._fail_or_default(default, Use.UnexpectedHash, f"The downloaded content of package {package_name} has a different hash than expected, aborting.")
                    try:
                        with open(path, "wb") as file:
                            file.write(download_response.content)
                        print("Downloaded", path)
                    except:
                        if fatal_exceptions: raise
                        exc = traceback.format_exc()
                    if exc:
                        return Use._fail_or_default(default, Use.AutoInstallationError, exc)
            
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
                        fileobj = open(tempfile.mkstemp()[0], "w")
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

                if exchange_sys_path:
                    temp_sys_path = copy(sys.path)
                    sys.path = [""]
                importlib.invalidate_caches()
                try:
                    sys.path.insert(0, "")
                    log.debug("Trying importlib.import_module")
                    log.debug("  with cwd=%s,", os.getcwd())
                    log.debug("  sys.path=%s", sys.path)
                    
                    mod = importlib.import_module(package_name)
                except ImportError:
                    if fatal_exceptions: raise
                    exc = traceback.format_exc()
                finally:
                    sys.path.remove("")
                    if exchange_sys_path:
                        sys.path = temp_sys_path
                if exc:
                    return Use._fail_or_default(default, ImportError, f"Failed to import {module_name} from {path}")

                for key in ("__name__", "__package__", "__path__", "__file__", "__version__", "__author__"):
                    if not hasattr(mod, key): continue
                    rdist_info[key] = getattr(mod, key)
                if not exc:
                    print(f"Successfully loaded {package_name}, version {version}.")
                os.chdir(original_cwd)
            
        self.persist_registry()
        for (check, pattern), decorator in aspectize.items():
            Use._apply_aspect(mod, check, pattern, decorator)
        self.set_mod(name=name, mod=mod, path=None, spec=spec, frame=inspect.getframeinfo(inspect.currentframe()))

        assert mod, f"Well shit. ( {path} )"
        return mod

sys.modules["use"] = Use()