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
- easy introspection via i

def test_db_setup(reuse):
      cur = reuse.registry.cursor()    nternal dependency graph (TODO)
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
>>> np = use("numpy", version="1.1.1", modes=use.auto_install, hash_value=["9879de676"])

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
import hashlib
import importlib.util
import inspect
import io
import json
import linecache
import os
import re
import shlex
import signal
import sqlite3
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
from functools import singledispatch, update_wrapper
from importlib import metadata
from logging import DEBUG, StreamHandler, getLogger, root
from pathlib import Path
from types import FrameType, ModuleType, TracebackType
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Union
from warnings import warn

import mmh3
import packaging
import requests
import toml
from furl import furl as URL
from packaging.specifiers import Specifier, SpecifierSet
from packaging.version import Version as PkgVersion


class PlatformTag(namedtuple("PlatformTag", ["platform"])):
    def __str__(self):
        return self.platform

    def __repr__(self):
        return self.platform


_supported = None


def get_supported() -> FrozenSet[PlatformTag]:
    global _supported
    if _supported is None:
        items: List[PlatformTag] = []
        try:
            from pip._internal.utils import compatibility_tags  # type: ignore

            for tag in compatibility_tags.get_supported():
                items.append(PlatformTag(platform=tag.platform))
        except ImportError:
            pass
        for tag in packaging.tags._platform_tags():
            items.append(PlatformTag(platform=str(tag)))
        _supported = tags = frozenset(items)
        log.error(str(tags))
    return _supported


_supported = None


def get_supported() -> FrozenSet[PlatformTag]:
    global _supported
    if _supported is None:
        items: List[PlatformTag] = []
        try:
            from pip._internal.utils import compatibility_tags  # type: ignore

            for tag in compatibility_tags.get_supported():
                items.append(PlatformTag(platform=tag.platform))
        except ImportError:
            pass
        for tag in packaging.tags._platform_tags():
            items.append(PlatformTag(platform=str(tag)))
        _supported = tags = frozenset(items)
        log.error(str(tags))
    return _supported


class VerHash(namedtuple("VerHash", ["version", "hash"])):
    @staticmethod
    def empty():
        return VerHash("", "")

    def __bool__(self):
        return bool(self.version and self.hash)

    def __eq__(self, other):
        if other is None or not hasattr(other, "__len__") or len(other) != len(self):
            return False
        return (
            Version(str(self.version)) == Version(str([*other][0]))
            and self.hash == [*other][1]
        )

    def __ne__(self, other):
        return not self.__eq__(other)


# injected via initial_globals for testing, you can safely ignore this
test_version = locals().get("test_version", None)
__version__ = test_version or "0.4.1"

_reloaders: Dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

# Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves.
# # This is necessary to compare sys.version_info with Version and make some tests more elegant, amongst other things.
class Version(PkgVersion):
    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        if isinstance(versionstr, str):
            return super().__init__(versionstr)
        else:
            return super().__init__(str(versionstr))  # this is just wrong :|

    def __iter__(self):
        yield from self.release


# Really looking forward to actual builtin sentinel values..
mode = Enum("Mode", "fastfail")

root.addHandler(StreamHandler(sys.stderr))
if "DEBUG" in os.environ:
    root.setLevel(DEBUG)
log = getLogger(__name__)

# defaults
config = {"version_warning": True, "debugging": False, "use_db": False}

# sometimes all you need is a sledge hammer..
def signal_handler(sig, frame):
    for reloader in _reloaders.values():
        reloader.stop()
    sig, frame
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# singledispatch for methods
def methdispatch(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


class ProxyModule(ModuleType):
    def __init__(self, mod):
        self.__implementation = mod
        self.__condition = threading.RLock()

    def __getattribute__(self, name):
        if name in ("_ProxyModule__implementation", "_ProxyModule__condition", ""):
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

    def start_async(self):
        loop = asyncio.get_running_loop()
        loop.create_task(self.run_async())

    def start_threaded(self):
        assert not (
            self._thread is not None and not self._thread.is_alive()
        ), "Can't start another reloader thread while one is already running."
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
            current_filehash = Use._hashfileobject(code)
            if current_filehash != last_filehash:
                try:
                    mod = Use._build_mod(
                        name=self.name,
                        code=code,
                        initial_globals=self.initial_globals,
                        module_path=self.path.resolve(),
                        aspectize=self.aspectize,
                    )
                    self.proxy.__implementation = mod
                except:
                    print(traceback.format_exc())
            last_filehash = current_filehash
            await asyncio.sleep(1)

    def run_threaded(self):
        last_filehash = None
        while not self._stopped:
            with self._condition:
                with open(self.path, "rb") as file:
                    code = file.read()
                current_filehash = Use._hashfileobject(code)
                if current_filehash != last_filehash:
                    try:
                        mod = Use._build_mod(
                            name=self.name,
                            code=code,
                            initial_globals=self.initial_globals,
                            module_path=self.path,
                            aspectize=self.aspectize,
                        )
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


# an instance of types.ModuleType
class Use(ModuleType):
    # lift module-level stuff up - ALIASES
    __doc__ = __doc__
    __version__ = __version__
    __name__ = __name__
    Version = Version

    # attempt at fix for #23 doesn't work..
    __path__ = str(Path(__file__).resolve().parent)

    class Hash(Enum):
        sha256 = hashlib.sha256

    ModInUse = namedtuple("ModInUse", "name mod path spec frame")

    # MODES to reduce signature complexity
    # enum.Flag wasn't really viable, but this is actually pretty cool
    auto_install = 2 ** 0
    fatal_exceptions = 2 ** 1
    reloading = 2 ** 2
    aspectize_dunders = 2 ** 3

    # ALIASES
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
        self.home: Path
        self._registry_dict = {}
        self._hacks = (
            {}
        )  # {(name -> interval_tree of Version -> function} basically plugins/workarounds for specific packages/versions

        self._set_up_files_and_directories()
        try:
            self._registry_db_connection = sqlite3.connect(self.home / "registry.db")
            self.registry = self._registry_db_connection.cursor()
        except:
            raise RuntimeError(
                "Could not connect to the registry database, please make sure it is accessible."
            )
        self._set_up_registry()
        assert self.registry is not None, "Registry is None"
        self._registry = Use._load_registry(self.home / "registry.json")
        self._user_registry = Use._load_registry(self.home / "user_registry.json")
        Use._merge_registry(self._registry, self._user_registry)

        # for the user to copy&paste
        with open(self.home / "default_config.toml", "w") as file:
            toml.dump(config, file)

        with open(self.home / "config.toml") as file:
            config.update(toml.load(file))

        if config["debugging"]:
            root.setLevel(DEBUG)

        if config["version_warning"]:
            try:
                response = requests.get("https://pypi.org/pypi/justuse/json")
                data = response.json()
                max_version = max(Version(version) for version in data["releases"].keys())
                if Version(__version__) < max_version:
                    warn(
                        f"""Justuse is version {Version(__version__)}, but there is a newer version {max_version} available on PyPI.
To find out more about the changes check out https://github.com/amogorkon/justuse/wiki/What's-new
Please consider upgrading via 'python -m pip install -U justuse'""",
                        Use.VersionWarning,
                    )
            except:
                log.debug(
                    traceback.format_exc()
                )  # we really don't need to bug the user about this (either pypi is down or internet is broken)

    # for easy refactoring later
    @property
    def _registry(self):
        return self._registry_dict

    @_registry.setter
    def _registry(self, value):
        self._registry_dict = value

    def _set_up_files_and_directories(self):
        self.home = Path.home() / ".justuse-python"
        try:
            self.home.mkdir(mode=0o755, parents=True, exist_ok=True)
        except PermissionError:
            # this should fix the permission issues on android #80
            self.home = Path(tempfile.mkdtemp(prefix="justuse_"))
        (self.home / "packages").mkdir(mode=0o755, parents=True, exist_ok=True)
        for file in (
            "registry.json",
            "user_registry.json",
            "config.toml",
            "config_defaults.toml",
            "usage.log",
            "registry.db",
            "user_registry.toml",
        ):
            (self.home / file).touch(mode=0o755, exist_ok=True)

    def _set_up_registry(self):
        self.registry.executescript(
            """
CREATE TABLE IF NOT EXISTS "distributions" (
	"id"	INTEGER,
	"name"	TEXT NOT NULL,
	"version"	TEXT NOT NULL,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "artifacts" (
	"id"	INTEGER,
	"distribution_id"	INTEGER,
	"tags"	TEXT,
	"url"	TEXT,
	"filename"	TEXT,
	"folder"	TEXT,
	"date_of_installation"	INTEGER,
	"number_of_uses"	INTEGER,
	"date_of_last_use"	INTEGER,
	FOREIGN KEY("distribution_id") REFERENCES "distributions"("id"),
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "hashes" (
	"algo"	TEXT NOT NULL,
	"value"	TEXT NOT NULL,
	"artifact_id"	INTEGER NOT NULL,
	PRIMARY KEY("algo","value"),
	FOREIGN KEY("artifact_id") REFERENCES "artifacts"("id")
);
        """
        )
        self.registry.connection.commit()

    def recreate_registry(self, use_db=False):
        if use_db:
            number_of_backups = len(list((self.home / "registry.db").glob("*.bak")))
            (self.home / "registry.json").rename(
                self.home / f"registry.json.{number_of_backups + 1}.bak"
            )
            (self.home / "registry.json").touch(mode=0o644)
            self._registry = Use._load_registry(self.home / "registry.db")
            self._user_registry = Use._load_registry(self.home / "user_registry.db")
            Use._merge_registry(self._registry_db, self._user_registry, use_db=True)
        else:
            number_of_backups = len(list((self.home / "registry.json").glob("*.bak")))
            (self.home / "registry.json").rename(
                self.home / f"registry.json.{number_of_backups + 1}.bak"
            )
            (self.home / "registry.json").touch(mode=0o644)
            self._registry = Use._load_registry(self.home / "registry.json")
            self._user_registry = Use._load_registry(self.home / "user_registry.json")
            Use._merge_registry(self._registry, self._user_registry)

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
        assert all(
            version is not None
            for version in (dist.keys() for dist in self._registry["distributions"].values())
        )

        with open(self.home / "registry.json", "w") as file:
            text = (
                "### WARNING: This file is automatically generated. Any manual changes will be overwritten. For persistent alterations, please use the user_registry.json ###\n"
                + json.dumps(self._registry, indent=2)
            )
            file.write(text)

    def register_hack(self, name, specifier=Specifier(">=0")):
        def wrapper(func):
            self._hacks[name] = func

        return wrapper

    def _registered_dist_names(self):
        return list(self._registry["distributions"].keys())

    def _versions_of_dist(self, name):
        return list(self._registry["distributions"][name].keys())

    def _dist_path(self, name, version):
        return Path(self._registry["distributions"][name][version]["path"])

    def _del_entry(self, name, version=None):
        if not version:
            del self._registry["distributions"][name]
        else:
            del self._registry["distributions"][name][version]

    def _entry_is_empty(self, name):
        return self._registry["distributions"][name] == {}

    def cleanup(self):
        """Bring registry and downloaded packages in sync.

        First all packages are removed that don't have a matching registry entry, then all registry entries that don't have a matching package.
        """

        def delete_folder(path):
            for sub in path.iterdir():
                if sub.is_dir():
                    delete_folder(sub)
                else:
                    sub.unlink()
            path.rmdir()

        # let's first take care of unregistered package folders
        for package_path in (self.home / "packages").iterdir():
            if package_path.stem not in (
                Path(version["path"]).stem
                for dist in self._registry["distributions"].values()
                for version in dist.values()
            ):
                if package_path.is_dir():
                    delete_folder(package_path)
                else:
                    package_path.unlink()

        # let's clean up the registry entries that don't have a matching package
        for name in self._registered_dist_names():
            for version in self._versions_of_dist(name):
                if not self._dist_path(name, version).exists():
                    self._del_entry(name, version)

        for name in self._registered_dist_names():
            if self._entry_is_empty(name):
                self._del_entry(name)
        self.persist_registry()

    def _save_module_info(
        self,
        name: str,
        version: Union[Version | str],  # type: ignore
        url: Optional[URL],
        path: Optional[Path],
        that_hash: Optional[str],
        folder: Path,
        package_name: str = None,
    ):
        """Update the registry to contain the package's metadata.
        Does not call Use.persist_registry() on its own."""
        package_name = package_name or name
        version = str(version) if version else "0.0.0"
        assert version not in ("None", "null", "")
        rdists = self._registry["distributions"]
        if package_name not in rdists:
            rdists[package_name] = {}
        if version not in rdists[package_name]:
            rdists[package_name][version] = {}
        assert url, "save_module_info received a missing URL"
        rdists[package_name][version].update(
            {
                "package": package_name,
                "version": version,
                "url": str(url),
                "path": str(path) if path else None,
                "folder": folder.absolute().as_uri(),
                "filename": path.name,
                "hash": that_hash,
            }
        )

    def _set_mod(self, *, name, mod, frame, path=None, spec=None):
        """Helper to get the order right."""
        self._using[name] = Use.ModInUse(name, mod, path, spec, frame)

    # hoisted functions - formerly module globals
    # staticmethods because module globals aren't reachable in tests while there should be no temptation for side effects via self

    @staticmethod
    def isfunction(x):
        return inspect.isfunction(x)

    @staticmethod
    def ismethod(x):
        return inspect.ismethod(x)

    # decorators for callable classes require a completely different approach i'm afraid.. removing the check should discourage users from trying
    # @staticmethod
    # def isclass(x):
    #     return inspect.isclass(x) and hasattr(x, "__call__")

    @staticmethod
    def _parse_filename(filename) -> dict:
        """Match the filename and return a dict of parts.
        >>> parse_filename("numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl")
        {'distribution': 'numpy', 'version': '1.19.5', 'build_tag', 'python_tag': 'cp36', 'abi_tag': 'cp36m', 'platform_tag': 'macosx_10_9_x86_64', 'ext': 'whl'}
        """
        assert isinstance(filename, str)
        match = re.match(
            "(?P<distribution>.*)-"
            "(?P<version>.*)"
            "(?:-(?P<build_tag>.*))?-"
            "(?P<python_tag>.*)-"
            "(?P<abi_tag>.*)-"
            "(?P<platform_tag>.*)\\."
            "(?P<ext>whl|zip|tar|egg|tar\\.gz)",
            filename,
        )
        return match.groupdict() if match else {}

    @staticmethod
    def _is_version_satisfied(info, sys_version):
        # https://warehouse.readthedocs.io/api-reference/json.html
        # https://packaging.pypa.io/en/latest/specifiers.html
        specifier = info.get(
            "requires_python", ""
        )  # SpecifierSet("") matches anything, no need to artificially lock down versions at this point
        # f.. you PyPI
        return sys_version in SpecifierSet(specifier or "")

    @staticmethod
    def _is_platform_compatible(
        info,
        platform_tags,
        include_sdist=False,
    ):
        assert isinstance(info, dict)
        assert isinstance(platform_tags, frozenset)
        # filename as API, seriously WTF...
        if "platform_tag" not in info:
            info.update(Use._parse_filename(info["filename"]))
        our_python_tag = "".join(
            (packaging.tags.interpreter_name(), packaging.tags.interpreter_version())
        )
        python_tag = info.get("python_tag", "")
        platform_tag = info.get("platform_tag", "")
        platform_srs = {*map(str, platform_tags)}
        for one_platform_tag in platform_tag.split("."):
            if one_platform_tag in platform_srs and our_python_tag == python_tag:
                log.info(
                    f"[Y] %s in {platform_srs=!r}, %s == %s",
                    one_platform_tag,
                    python_tag,
                    our_python_tag,
                )
                return True
        return include_sdist and info["filename"].endswith(".egg")

    @staticmethod
    def _find_matching_artifact(
        urls: List[Dict[str, Any]],
        hash_algo=Hash.sha256.name,
        *,
        # for testability
        sys_version=None,
        platform_tags=frozenset(),
        interpreter_tag=None,
        include_sdist=False,
    ) -> VerHash:
        """Pick from a list of possible urls and return the
        `(version, hash_value)`
        pair from the best-fitting artifact,
        preferring bdist over sdist distribution archives."""
        info_s = sorted(urls, key=lambda i: i.get("packagetype", ""))
        for include_sdist in (False, True):  # prefer non-source
            results = [
                VerHash(info["version"], info["digests"][hash_algo])
                for info in info_s
                if Use._is_compatible(
                    info, hash_algo, sys_version, platform_tags, include_sdist
                )
            ]
            if results:
                return results[0]
        return VerHash.empty()

    @staticmethod
    def _is_compatible(
        info,
        hash_algo=Hash.sha256.name,
        sys_version=None,
        platform_tags=frozenset(),
        include_sdist=False,
    ):
        """Return true if the artifact described by 'info'
        is compatible with the current or specified system."""
        assert isinstance(info, dict)
        if "platform_tag" not in info:
            info.update(Use._parse_filename(info["filename"]))
        sys_version = sys_version or Version(".".join(map(str, sys.version_info[0:3])))
        assert isinstance(sys_version, Version)
        platform_tags = platform_tags or get_supported()
        return ".egg" not in info["filename"] and (
            include_sdist
            or (
                Use._is_version_satisfied(info, sys_version)
                and Use._is_platform_compatible(info, platform_tags, include_sdist)
                and not info["yanked"]
            )
        )

    @staticmethod
    def _find_latest_working_version(
        releases: Dict[
            str, List[Dict[str, Any]]
        ],  # {version: [{comment_text, filename, url, version, hash, build_tag, python_tag, abi_tag, platform_tag: str}]}
        *,
        hash_algo,
        # testing
        sys_version=None,
        platform_tags=frozenset(),
        interpreter_tag=None,
        version=None,
    ) -> VerHash:
        assert isinstance(releases, dict)
        assert isinstance(hash_algo, str)
        # update the release dicts to hold all info canonically
        # be aware, the json returned from pypi ["releases"] can
        # contain empty lists as dists :/
        for ver, dists in releases.items():
            if not dists:
                continue
            for d in dists:
                d["version"] = ver  # Add version info
                d.update(Use._parse_filename(d["filename"]))
        for include_sdist in (False, True):  # prefer non-source
            for ver, infos in sorted(
                releases.items(), key=lambda item: Version(item[0]), reverse=True
            ):
                for info in dists:
                    if Use._is_compatible(
                        info, hash_algo, sys_version, platform_tags, include_sdist
                    ):
                        return VerHash(info["version"], info["digests"][hash_algo])
        return VerHash.empty()

    @staticmethod
    def _load_registry(path):
        registry_version = "0.0.2"
        registry = {}
        with open(path) as file:
            # json doesn't have comments, so we need to manually skip the first line warning for the user
            lines = file.readlines()
            if not lines:  # might be an empty file
                registry.update(
                    {
                        "version": registry_version,
                        "distributions": defaultdict(lambda: dict()),
                    }
                )
                return registry
            registry.update(
                json.loads("\n".join(filter(lambda s: not s.startswith("#"), lines)))
            )  # Now comments in user_registry.json are ignored, too

        if "version" in registry and registry["version"] < registry_version:
            print(
                f"Registry is being upgraded from version {registry.get('version',0)} to version {registry_version}"
            )
            registry["version"] = registry_version
        elif registry and "version" not in registry:
            print("Registry is being upgraded from version 0")
            new_registry = {
                "version": registry_version,
                "distributions": (dists := defaultdict(lambda: dict())),
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
        buf = b""
        while True:
            towrite = number & 0x7F
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
        return enc_size + hash_[len(enc_size) :]

    @staticmethod
    def _build_mod(
        *,
        name,
        code,
        initial_globals: Optional[Dict[str, Any]],
        module_path,
        aspectize,
        default=mode.fastfail,
        aspectize_dunders=aspectize_dunders,
        package=None,
    ) -> ModuleType:
        default
        mod = ModuleType(name)
        mod.__dict__.update(initial_globals or {})
        mod.__file__ = str(module_path)
        code_text = codecs.decode(code)
        # module file "<", ">" chars are specially handled by inspect
        if not sys.platform.startswith("win"):
            getattr(linecache, "cache")[f"<{name}>"] = (
                len(code),  # size of source code
                None,  # last modified time; None means there is no physical file
                [
                    *map(  # a list of lines, including trailing newline on each
                        lambda ln: ln + "\x0a", code_text.splitlines()
                    )
                ],
                mod.__file__,  # file name, e.g. "<mymodule>" or the actual path to the file
            )
        # not catching this causes the most irritating bugs ever!
        if package:
            mod.__package__ = package
        try:
            exec(compile(code, f"<{name}>", "exec"), mod.__dict__)
        except:  # reraise anything without handling - clean and simple.
            raise
        for (check, pattern), decorator in aspectize.items():
            Use._apply_aspect(
                mod, check, pattern, decorator, aspectize_dunders=aspectize_dunders
            )
        return mod

    @staticmethod
    def _fail_or_default(default, exception, msg):
        if default is not Use.mode.fastfail:
            return default
        else:
            raise exception(msg)

    @staticmethod
    def _apply_aspect(
        thing,
        check,
        pattern,
        decorator: Callable[[Callable[..., Any]], Any],
        aspectize_dunders=False,
    ):
        """Apply the aspect as a side-effect, no copy is created."""
        for name, obj in thing.__dict__.items():
            if not aspectize_dunders and name.startswith("__") and name.endswith("__"):
                continue
            if check(obj) and re.match(pattern, name):
                log.debug(f"Applying aspect to {thing}.{name}")
                thing.__dict__[name] = decorator(obj)
        return thing

    @methdispatch
    def __call__(self, thing, /, *args, **kwargs):
        raise NotImplementedError(
            f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}."
        )

    @__call__.register(URL)
    def _use_url(
        self,
        url: URL,
        /,
        *,
        hash_algo=Hash.sha256,
        hash_value=None,
        initial_globals: Optional[Dict[Any, Any]] = None,
        as_import=None,
        default=mode.fastfail,
        aspectize=None,
        path_to_url=None,
        import_to_use=None,
        modes=0,
    ) -> ModuleType:
        exc = None

        assert hash_algo in Use.Hash, f"{hash_algo} is not a valid hashing algorithm!"

        aspectize = aspectize or {}
        response = requests.get(str(url))
        if response.status_code != 200:
            raise ImportError(
                f"Could not load {url} from the interwebs, got a {response.status_code} error."
            )
        this_hash = hash_algo.value(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                return Use._fail_or_default(
                    default,
                    Use.UnexpectedHash,
                    f"{this_hash} does not match the expected hash {hash_value} - aborting!",
                )
        else:
            warn(
                f"""Attempting to import from the interwebs with no validation whatsoever!
To safely reproduce: use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')""",
                Use.NoValidationWarning,
            )
        name = str(url)

        try:
            mod = Use._build_mod(
                name=name,
                code=response.content,
                module_path=url.path,
                initial_globals=initial_globals,
                aspectize=aspectize,
                aspectize_dunders=bool(Use.aspectize_dunders & modes),
            )
        except:
            exc = traceback.format_exc()
        if exc:
            return Use._fail_or_default(default, ImportError, exc)

        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        if as_import:
            assert isinstance(
                as_import, str
            ), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            assert as_import.isidentifier(), "as_import must be a valid identifier."
            sys.modules[as_import] = mod
        return mod

    @__call__.register(Path)
    def _use_path(
        self,
        path,
        /,
        *,
        initial_globals=None,
        as_import=None,
        default=mode.fastfail,
        aspectize=None,
        path_to_url=None,
        import_to_use=None,
        modes=0,
    ) -> Optional[ModuleType]:
        aspectize = aspectize or {}
        initial_globals = initial_globals or {}

        reloading = bool(Use.reloading & modes)

        exc = None
        mod = None

        if path.is_dir():
            return Use._fail_or_default(
                default, ImportError, f"Can't import directory {path}"
            )

        original_cwd = source_dir = Path.cwd()
        try:
            if not path.is_absolute():
                source_dir = getattr(
                    self._using.get(inspect.currentframe().f_back.f_back.f_code.co_filename),
                    "path",
                    None,
                )

            # calling from another use()d module
            if source_dir and source_dir.exists():
                os.chdir(source_dir.parent)
                source_dir = source_dir.parent
            else:
                # there are a number of ways to call use() from a non-use() starting point
                # let's first check if we are running in jupyter
                jupyter = "ipykernel" in sys.modules
                # we're in jupyter, we use the CWD as set in the notebook
                if not jupyter:
                    # let's see where we started
                    main_mod = __import__("__main__")
                    # if we're calling from a script file e.g. `python3 my/script.py` like pytest unittest
                    if hasattr(main_mod, "__file__"):
                        source_dir = (
                            Path(inspect.currentframe().f_back.f_back.f_code.co_filename)
                            .resolve()
                            .parent
                        )
            if not source_dir.exists():
                return Use._fail_or_default(
                    default,
                    NotImplementedError,
                    "Can't determine a relative path from a virtual file.",
                )
            path = source_dir.joinpath(path).resolve()
            if not path.exists():
                return Use._fail_or_default(default, ImportError, f"Sure '{path}' exists?")
            os.chdir(path.parent)
            name = path.stem
            if reloading:
                try:
                    with open(path, "rb") as file:
                        code = file.read()
                    # initial instance, if this doesn't work, just throw the towel
                    mod = Use._build_mod(
                        name=name,
                        code=code,
                        initial_globals=initial_globals,
                        module_path=str(path.resolve()),
                        aspectize=aspectize,
                    )
                except:
                    exc = traceback.format_exc()
                if exc:
                    return Use._fail_or_default(default, ImportError, exc)
                mod = ProxyModule(mod)
                reloader = ModuleReloader(
                    proxy=mod,
                    name=name,
                    path=path,
                    initial_globals=initial_globals,
                    aspectize=aspectize,
                )
                _reloaders[mod] = reloader

                threaded = False
                # this looks like a hack, but isn't one -
                # jupyter is running an async loop internally, which works better async than threaded!
                try:
                    asyncio.get_running_loop()
                # we're dealing with non-async code, we need threading
                except RuntimeError:
                    # can't have the code inside the handler because of "during handling of X, another exception Y happened"
                    threaded = True
                if not threaded:
                    reloader.start_async()
                else:
                    reloader.start_threaded()

                if not all(
                    inspect.isfunction(value)
                    for key, value in mod.__dict__.items()
                    if key not in initial_globals.keys() and not key.startswith("__")
                ):
                    warn(
                        f"Beware {name} also contains non-function objects, it may not be safe to reload!",
                        Use.NotReloadableWarning,
                    )
            else:  # NOT reloading
                with open(path, "rb") as file:
                    code = file.read()
                # the path needs to be set before attempting to load the new module - recursion confusing ftw!
                frame = inspect.getframeinfo(inspect.currentframe())
                self._set_mod(name=name, mod=mod, frame=frame)
                try:
                    mod = Use._build_mod(
                        name=name,
                        code=code,
                        initial_globals=initial_globals,
                        module_path=str(path),
                        aspectize=aspectize,
                    )
                except:
                    del self._using[f"<{name}>"]
                    exc = traceback.format_exc()
        except:
            exc = traceback.format_exc()
            return Use._fail_or_default(default, ImportError, exc)
        finally:
            # let's not confuse the user and restore the cwd to the original in any case
            os.chdir(original_cwd)
        if exc:
            return Use._fail_or_default(default, ImportError, exc)
        if as_import:
            assert isinstance(
                as_import, str
            ), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            sys.modules[as_import] = mod
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return mod

    def _import_builtin(self, name, spec, default, aspectize):
        try:
            mod = spec.loader.create_module(spec)
            spec.loader.exec_module(mod)  # ! => cache
            if aspectize:
                warn(
                    "Applying aspects to builtins may lead to unexpected behaviour, but there you go..",
                    RuntimeWarning,
                )
            for (check, pattern), decorator in aspectize.items():
                Use._apply_aspect(
                    mod, check, pattern, decorator, aspectize_dunders=self.aspectize_dunders
                )
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=mod, spec=spec, frame=frame)
            return mod
        except:
            exc = traceback.format_exc()
            return Use._fail_or_default(default, ImportError, exc)

    @staticmethod
    def _get_version(name=None, package_name=None, /, mod=None) -> Optional[Version]:
        assert name is None or isinstance(name, str)
        version = None
        for lookup_name in (name, package_name):
            if not lookup_name:
                continue
            try:
                if lookup_name is not None:
                    meta = metadata.distribution(lookup_name)
                    return Version(meta.version)
            except metadata.PackageNotFoundError:
                continue
        if not mod:
            return None
        version = getattr(mod, "__version__", version)
        if isinstance(version, str):
            return Version(version)
        version = getattr(mod, "version", version)
        if callable(version):
            vevsion = version()
        if isinstance(version, str):
            return Version(version)
        return version

    def _import_classical_install(
        self,
        name,
        module_name,
        spec,
        target_version,
        default,
        aspectize,
        fatal_exceptions,
        package_name=None,
    ):
        # sourcery no-metrics
        exc = None
        try:
            mod = importlib.import_module(module_name)  # ! => cache
            for (check, pattern), decorator in aspectize.items():
                Use._apply_aspect(
                    mod,
                    check,
                    pattern,
                    decorator,
                    aspectize_dunders=bool(Use.aspectize_dunders & self.modes),
                )
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=mod, frame=frame)
            if not target_version:
                warn(
                    f"Classically imported '{name}'. To pin this version use('{name}', version='{metadata.version(name)}')",
                    Use.AmbiguityWarning,
                )
        except:
            if fatal_exceptions:
                raise
            exc = traceback.format_exc()
        if exc:
            return Use._fail_or_default(default, ImportError, exc)
        # we only enforce versions with auto-install
        this_version = Use._get_version(name, package_name, mod=mod)
        if not this_version:
            log.warning(f"Cannot find version for {name=}, {mod=}")
        elif not target_version:
            warn("No version was specified", Use.AmbiguityWarning)
        elif target_version != this_version:
            warn(
                f"{name} expected to be version {target_version},"
                f" but got {this_version} instead",
                Use.VersionWarning,
            )
        for (check, pattern), decorator in aspectize.items():
            Use._apply_aspect(
                mod,
                check,
                pattern,
                decorator,
                aspectize_dunders=bool(Use.aspectize_dunders & self.modes),
            )
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return mod

    @__call__.register(str)
    def _use_str(
        self,
        name,
        /,
        *,
        version=None,
        initial_globals=None,
        hash_algo=Hash.sha256,
        hash_value=None,
        hash_values=None,
        default=mode.fastfail,
        aspectize=None,
        path_to_url=None,
        import_to_use=None,
        modes=0,
        fatal_exceptions=True,
        package_name=None,  # internal use
        module_name=None,  # internal use
    ) -> Optional[ModuleType]:

        self.modes = modes
        initial_globals = initial_globals or {}
        aspectize = aspectize or {}
        path = None
        hash_values = hash_values or []
        if hash_value:
            if isinstance(hash_value, str):
                hash_values += shlex.split(hash_value)
            else:
                hash_values += list(hash_value)
        elif hash_values:
            hash_value = " ".join(hash_values)
        # we use boolean flags to reduce the complexity of the call signature
        fatal_exceptions = bool(Use.fatal_exceptions & modes)
        auto_install = bool(Use.auto_install & modes)

        # the whole auto-install shebang
        package_name, _, module_name = name.partition(".")
        module_name = module_name or name

        assert (
            version is None or isinstance(version, str) or isinstance(version, Version)
        ), "Version must be given as string or packaging.version.Version."
        target_version = Version(str(version)) if version else None
        # just validating user input and canonicalizing it
        version = str(target_version) if target_version else None
        assert (
            version if target_version else version is target_version
        ), "Version must be None if target_version is None; otherwise, they must both have a value."
        exc = None
        mod = None

        if initial_globals or import_to_use or path_to_url:
            raise NotImplementedError(
                "If you require this functionality, please report it on https://github.com/amogorkon/justuse/issues so we can work out the specifics together."
            )

        # The "try and guess" behaviour is due to how classical imports work,
        # which is inherently ambiguous, but can't really be avoided for packages.
        # let's first see if the user might mean something else entirely
        if any(Path(".").glob(f"{name}.py")):
            warn(
                f"Attempting to load the package '{name}', if you rather want to use the local module: use(use.Path('{name}.py'))",
                Use.AmbiguityWarning,
            )
        hit: VerHash = VerHash.empty()
        data = None
        spec = None
        entry = None
        found = None
        that_hash = None
        all_that_hash = []
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
                return self._import_builtin(name, spec, default, aspectize)

            # it seems to be installed in some way, for instance via pip
            if not auto_install:
                return self._import_classical_install(
                    name,
                    module_name,
                    spec,
                    target_version,
                    default,
                    aspectize,
                    fatal_exceptions,
                )

            this_version = Use._get_version(name, package_name)

            if this_version == target_version:
                if not (version):
                    warn(
                        Use.AmbiguityWarning(
                            "No version was provided, even though auto_install was specified! Trying to load classically installed package instead."
                        )
                    )
                mod = self._import_classical_install(
                    name,
                    module_name,
                    spec,
                    target_version,
                    default,
                    aspectize,
                    fatal_exceptions,
                    package_name,
                )
                warn(
                    f'Classically imported \'{name}\'. To pin this version: use("{name}", version="{this_version}")',
                    Use.AmbiguityWarning,
                )
                return mod
            elif not (version):
                warn(
                    Use.AmbiguityWarning(
                        "No version was provided, even though auto_install was specified! Trying to load classically installed package instead."
                    )
                )
                mod = self._import_classical_install(
                    name,
                    module_name,
                    spec,
                    target_version,
                    default,
                    aspectize,
                    fatal_exceptions,
                    package_name,
                )
                warn(
                    f'Classically imported \'{name}\'. To pin this version: use("{name}", version="{this_version}")',
                    Use.AmbiguityWarning,
                )
                return mod
            # wrong version => wrong spec
            this_version = Use._get_version(mod=mod)
            if this_version != target_version:
                spec = None
                log.warning(
                    f"Setting {spec=}, since " f"{target_version=} != {this_version=}"
                )
        else:
            if not auto_install:
                return Use._fail_or_default(
                    default,
                    ImportError,
                    f"Could not find any installed package '{name}' and auto_install was not requested.",
                )
            # PEBKAC
            hit: VerHash = VerHash.empty()
            if target_version and not (hash_value or hash_values):  # let's try to be helpful
                response = requests.get(
                    f"https://pypi.org/pypi/{package_name}/{target_version}/json"
                )
                if response.status_code == 404:
                    raise RuntimeWarning(
                        f"Are you sure {package_name} with version {version} exists?"
                    )
                elif response.status_code != 200:
                    raise RuntimeWarning(
                        f"Something bad happened while contacting PyPI for info on {package_name} ( {response.status_code} ), which we tried to look up because a matching hash_value for the auto-installation was missing."
                    )
                data = response.json()

                version, that_hash = hit = Use._find_matching_artifact(
                    data["urls"], hash_algo=hash_algo.name
                )
                log.info(f"{hit=} from  Use._find_matching_artifact")
                if that_hash:
                    if that_hash is not None:
                        hash_value = that_hash
                    if that_hash is not None:
                        hash_values = hash_values + [that_hash]

                    if that_hash is not None:
                        hash_value = that_hash
                    if that_hash is not None:
                        hash_values = hash_values + [that_hash]

                    raise RuntimeWarning(
                        f"""Failed to auto-install '{package_name}' because hash_value is missing. This may work:
use("{package_name}", version="{version}", hash_value="{that_hash}", modes=use.auto_install)
"""
                    )
                raise RuntimeWarning(
                    f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
                )
            elif not target_version and (hash_value or hash_values):
                raise RuntimeWarning(
                    f"Failed to auto-install '{package_name}' because no version was specified."
                )
            elif not target_version:
                # let's try to make an educated guess and give a useful suggestion
                response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
                if response.status_code == 404:
                    # possibly typo - PEBKAC
                    raise RuntimeWarning(
                        f"Are you sure package '{package_name}' "
                        "exists? Could not find any package with "
                        "that name on PyPI."
                    )
                elif response.status_code != 200:
                    # possibly server problems
                    return Use._fail_or_default(
                        default,
                        Use.AutoInstallationError,
                        f"Tried to look up '{package_name}', but "
                        "got a {response.status_code} from PyPI.",
                    )

                data = response.json()
                version, hash_value = hit = Use._find_latest_working_version(
                    data["releases"], hash_algo=hash_algo.name
                )

                if not hash_value:
                    raise RuntimeWarning(
                        f"We could not find any version or release "
                        f"for {package_name} that could satisfy our "
                        f"requirements!"
                    )

                if not target_version and (hash_value or hash_values):
                    hash_values += [hash_value]
                    raise RuntimeWarning(
                        f"""Please specify version and hash for auto-installation of '{package_name}'.
To get some valuable insight on the health of this package, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version: use("{name}", version="{version}", hash_value="{hash_value}", modes=use.auto_install)
"""
                    )

            # all clear, let's check if we pulled it before
            entry = entry or self._registry["distributions"].get(package_name, {}).get(
                version, {}
            )
            if entry and entry["path"]:
                path = Path(entry["path"])
                url = URL(entry["url"])
            if entry and path and not path.exists():
                del self._registry["distributions"][package_name][version]
                self.persist_registry()
                path = None
                entry = None
            if not path:
                response = requests.get(
                    f"https://pypi.org/pypi/{package_name}/{target_version}/json"
                )
                if response.status_code != 200:
                    return Use._fail_or_default(
                        default,
                        ImportError,
                        f"Tried to auto-install '{package_name}' {target_version} but failed with {response} while trying to pull info from PyPI.",
                    )
                try:
                    data = response.json()
                    if not "urls" in data:
                        return Use._fail_or_default(
                            default,
                            Use.AutoInstallationError,
                            f"Tried to auto-install {package_name} {target_version} but failed because no valid URLs to download could be found.",
                        )
                    for entry in data["urls"]:
                        url = URL(entry["url"])
                        entry["version"] = str(target_version)
                        log.debug(f"looking at {entry=}")
                        all_that_hash.append(
                            that_hash := entry["digests"].get(hash_algo.name)
                        )
                        if (not hash_value or not hash_values) or (
                            hash_value in all_that_hash
                            or len(hash_values) > 1
                            and set(hash_values).intersection(set(all_that_hash))
                        ):
                            found = (entry, url, that_hash)
                            hit = VerHash(version, that_hash)
                            log.info(f"Matchrs user hash: {entry=} {hit=}")
                            break
                    if found is None:
                        return Use._fail_or_default(
                            default,
                            Use.AutoInstallationError,
                            f"Tried to auto-install {name!r} ({package_name=!r}) with {target_version=!r} but failed because none of the available hashes ({all_that_hash=!r}) match the expected hash ({hash_value=!r} or {hash_values=!r}).",
                        )
                    entry, url, that_hash = found
                    hash_value = that_hash
                    if that_hash is not None:
                        hash_values = [that_hash]
                except BaseException as be:  # json issues
                    msg = (
                        f"request to "
                        f"https://pypi.org/pypi/{package_name}/{target_version}/json"
                        f"lead to an error: {be}"
                    )
                    raise RuntimeError(msg, response) from be
                    # exc = traceback.format_exc()
                if exc:
                    return Use._fail_or_default(
                        default,
                        Use.AutoInstallationError,
                        f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI.",
                    )
                # we've got a complete JSON with a matching entry, let's download
                path = (
                    self.home / "packages" / Path(url.asdict()["path"]["segments"][-1]).name
                )
                if not path.exists():
                    print("Downloading", url, "...")
                    download_response = requests.get(str(url), allow_redirects=True)
                    path = (
                        self.home
                        / "packages"
                        / Path(url.asdict()["path"]["segments"][-1]).name
                    )
                    this_hash = hash_algo.value(download_response.content).hexdigest()
                    if this_hash != hash_value:
                        return Use._fail_or_default(
                            default,
                            Use.UnexpectedHash,
                            f"The downloaded content of package {package_name} has a different hash than expected, aborting.",
                        )
                    try:
                        with open(path, "wb") as file:
                            file.write(download_response.content)
                        print("Downloaded", path)
                    except:
                        if fatal_exceptions:
                            raise
                        exc = traceback.format_exc()
                    if exc:
                        return Use._fail_or_default(default, Use.AutoInstallationError, exc)

            # now that we can be sure we got a valid package downloaded and ready, let's try to install it
            folder = path.parent / path.stem
            rdists = self._registry["distributions"]
            if not url:
                url = URL(f"file:/{path}")
            if name in self._hacks:
                # if version in self._hacks[name]:
                mod = self._hacks[name](
                    package_name=package_name,
                    rdists=rdists,
                    version=Use._get_version(mod=mod),
                    url=url,
                    path=path,
                    that_hash=hash_value,
                    folder=folder,
                    fatal_exceptions=fatal_exceptions,
                    module_name=module_name,
                )
                return mod

            # trying to import directly from zip
            try:
                importer = zipimport.zipimporter(path)
                mod = importer.load_module(module_name)
                print("Direct zipimport of", name, "successful.")
            except:
                if config["debugging"]:
                    log.debug(traceback.format_exc())
                return self._fail_or_default(
                    default,
                    Use.AutoInstallationError,
                    f"Direct zipimport of {name} {version} failed and the package was not registered with known hacks.. we're sorry, but that means you will need to resort to using pip/conda for now.",
                )
        self.persist_registry()
        for (check, pattern), decorator in aspectize.items():
            if mod is not None:
                Use._apply_aspect(
                    mod,
                    check,
                    pattern,
                    decorator,
                    aspectize_dunders=bool(Use.aspectize_dunders & modes),
                )
        frame = inspect.getframeinfo(inspect.currentframe())
        if frame:
            self._set_mod(name=name, mod=mod, frame=frame)
        assert mod, f"Well. Shit, no module. ( {path} )"
        this_version = Use._get_version(mod=mod) or version
        assert this_version, f"Well. Shit, no version. ( {path} )"
        self._save_module_info(
            name, this_version, url, path, that_hash, folder, package_name=package_name
        )
        self.persist_registry()
        return mod

    pass


# we should avoid side-effects during testing, specifically for the version-upgrade-warning
Use.Version = Version
Use.config = config
Use.mode = mode
Use.Path = Path
Use.URL = URL
use = Use()
if not test_version:
    sys.modules["use"] = use

# no circular import this way
hacks_path = Path(Path(__file__).parent, "package_hacks.py")
assert hacks_path.exists()
use(hacks_path)  # type: ignore
