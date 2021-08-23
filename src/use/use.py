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
                    hashes="95f98f25ef8cfa0102642ea5babbe6dde3e3a19d411db9164af53a9b4cdcccd8")

# to auto-install a certain version (within a virtual env and pip in secure hash-check mode) of a package you can do
>>> np = use("numpy", version="1.1.1", modes=use.auto_install, hash_value=["9879de676"])

File-Hashing inspired by
- https://github.com/kalafut/py-imohash
- https://github.com/fmoo/python-varint/blob/master/varint.py

:author: Anselm Kiefner (amogorkon)
:author: David Reilly
:license: MIT
"""


from __future__ import annotations

import asyncio
import atexit
import codecs
import functools
import hashlib
import importlib.util
import inspect
import linecache
import operator
import os
import re
import shlex
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import zipimport
import operator

from collections import defaultdict, namedtuple
from enum import Enum
from functools import singledispatch, update_wrapper
from importlib import metadata
from importlib.machinery import SourceFileLoader
from logging import DEBUG, NOTSET, WARN, StreamHandler, getLogger, root
from pathlib import Path
from pkgutil import zipimporter
from subprocess import check_output, run
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
root.setLevel(NOTSET)
if "DEBUG" in os.environ:
    root.setLevel(DEBUG)
else:
    root.setLevel(WARN)

# TODO: log to file
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
    __name__ = __name__
    __file__ = __file__
    __spec__ = __spec__
    Use = property(lambda _: Use)

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

    @staticmethod
    def dict_factory(cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    def __init__(self):
        self._using = _using
        self._aspects = _aspects
        self._reloaders = _reloaders
        self.home: Path
        self._hacks = (
            {}
        )  # {(name -> interval_tree of Version -> function} basically plugins/workarounds for specific packages/versions

        self._set_up_files_and_directories()
        # might run into issues during testing otherwise
        if not test_version:
            try:
                self.registry = sqlite3.connect(self.home / "registry.db").cursor()
                self.registry.execute("PRAGMA foreign_keys=ON")
                self.registry.execute("PRAGMA auto_vacuum = FULL")
            except Exception as e:
                raise RuntimeError(
                    f"Could not connect to the registry database, please make sure it is accessible. ({e})"
                )
        else:
            self.registry = sqlite3.connect(":memory:").cursor()
        self.registry.row_factory = Use.dict_factory

        self._set_up_registry()
        self._user_registry = toml.load(self.home / "user_registry.toml")

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

    def _set_up_files_and_directories(self):
        self.home = Path.home() / ".justuse-python"
        try:
            self.home.mkdir(mode=0o755, parents=True, exist_ok=True)
        except PermissionError:
            # this should fix the permission issues on android #80
            self.home = Path(tempfile.mkdtemp(prefix="justuse_"))
        (self.home / "packages").mkdir(mode=0o755, parents=True, exist_ok=True)
        for file in (
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
CREATE TABLE IF NOT EXISTS "artifacts" (
	"id"	INTEGER,
	"distribution_id"	INTEGER,
	"path"	TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("distribution_id") REFERENCES "distributions"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "distributions" (
	"id"	INTEGER,
	"name"	TEXT NOT NULL,
	"version"	TEXT NOT NULL,
	"installation_path"	TEXT,
	"date_of_installation"	INTEGER,
	"number_of_uses"	INTEGER,
	"date_of_last_use"	INTEGER,
	"pure_python_package"	INTEGER NOT NULL DEFAULT 1,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "hashes" (
	"algo"	TEXT NOT NULL,
	"value"	TEXT NOT NULL,
	"artifact_id"	INTEGER NOT NULL,
	PRIMARY KEY("algo","value"),
	FOREIGN KEY("artifact_id") REFERENCES "artifacts"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "depends_on" (
	"origin_path"	TEXT,
	"target_path"	TEXT
, "time_of_use"	INTEGER)
        """
        )
        self.registry.connection.commit()

    def recreate_registry(self):
        number_of_backups = len(list((self.home / "registry.db").glob("*.bak")))
        (self.home / "registry.db").rename(
            self.home / f"registry.db.{number_of_backups + 1}.bak"
        )
        (self.home / "registry.db").touch(mode=0o644)
        self._set_up_registry()

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

    def del_entry(self, name, version):
        # TODO: CASCADE to artifacts etc
        self.registry.execute(
            "DELETE FROM distributions WHERE name=? AND version=?", (name, version)
        )
        self.registry.connection.commit()

    def register_hack(self, name, specifier=Specifier(">=0")):
        def wrapper(func):
            self._hacks[name] = func

        return wrapper

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

        installation_paths = self.registry.execute(
            "SELECT installation_path FROM distributions"
        ).fetchall()
        for package_path in (self.home / "packages").iterdir():
            if package_path.stem not in installation_paths:
                if package_path.is_dir():
                    delete_folder(package_path)
                else:
                    package_path.unlink()

        for ID, path in self.registry.execute(
            "SELECT id, installation_path FROM distributions"
        ).fetchall():
            if not Path(path).exists():
                self.registry.execute(f"DELETE FROM distributions WHERE id=?", (ID,))
        self.registry.connection.commit()

    def _save_module_info(
        self,
        *,
        version: Union[Version | str],  # type: ignore
        artifact_path: Optional[Path],
        hash_value=Optional[str],
        installation_path=Path,
        name: str,
        hash_algo=Hash.sha256,
    ):
        """Update the registry to contain the package's metadata.
        Does not call Use.persist_registry() on its own."""
        version = str(version) if version else "0.0.0"
        assert version not in ("None", "null", "")

        if not self.registry.execute(
            f"SELECT * FROM distributions WHERE name='{name}' AND version='{version}'"
        ).fetchone():
            self.registry.execute(
                f"""
INSERT INTO distributions (name, version, installation_path, date_of_installation, pure_python_package)
VALUES ('{name}', '{version}', '{installation_path}', {time.time()}, {installation_path is None})
"""
            )
            self.registry.execute(
                f"""
INSERT OR IGNORE INTO artifacts (distribution_id, path)
VALUES ({self.registry.lastrowid}, '{artifact_path}')
"""
            )
            self.registry.execute(
                f""" 
INSERT OR IGNORE INTO hashes (artifact_id, algo, value)
VALUES ({self.registry.lastrowid}, '{hash_algo.name}', '{hash_value}')"""
            )
        self.registry.connection.commit()

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
    def _load_venv_mod(package, version):
        venv_root = Use._venv_root(package, version)
        venv_bin = venv_root / "bin"
        python_exe = Path(sys.executable).stem
        if not venv_bin.exists():
             check_output(
                [
                    python_exe, "-m", "venv",
                    "--system-site-packages",
                    venv_root
                ],
                encoding="UTF-8"
             )
        pip_args = (
            python_exe, "-m", "pip",
            "--no-python-version-warning",
            "--disable-pip-version-check",
            "--no-color",
            "install",
            "--progress-bar",
            "ascii",
            "--prefer-binary",
            "--exists-action", "b",
            "--only-binary", ":all:",
            "--no-build-isolation",
            "--no-use-pep517",
            "--no-compile",
            "--no-warn-script-location",
            "--no-warn-conflicts",
            f"{package}=={version}",
        )
        current_path = os.environ.get("PATH")
        venv_path_var = f"{venv_bin}{os.path.pathsep}{current_path}"
        pkg_path = Use._venv_pkg_path(package, version)
        if Use._venv_is_win():
            output = run(
                [
                  "cmd.exe", "/C",
                  "set", f"PATH={venv_path_var}", "&",
                  *pip_args
                ],
                encoding="UTF-8",
                stderr=sys.stderr
            )
        else:
            output = run(
                ["env", f"PATH={venv_path_var}", *pip_args],
                encoding="UTF-8",
                stderr=sys.stderr
            )
        log.debug("pip subprocess output=%r", output)
        orig_cwd = Path.cwd()
        try:
            os.chdir(pkg_path)
            pkg_prefix, entry_module_path = Use._find_entry_point(
                package, version
            )
            path = entry_module_path.absolute()
            with open(path, "rb") as f:
                code = f.read()
                Use._clean_sys_modules(package)
                Use._clean_sys_modules(pkg_prefix)
                Use._clean_sys_modules(f"{pkg_prefix}.{package}")
                mod = use._build_mod(
                    name=package,
                    code=code,
                    module_path=path,
                    initial_globals={},
                    aspectize={},
                    package=(pkg_prefix or package)
                )
                log.debug(
                  f"module returned from _load_venv_mod: {mod}"
                )
                return mod
        finally:
            os.chdir(orig_cwd)

    @staticmethod
    def _venv_pkg_path(package, version):
        venv_root = Use._venv_root(package, version)
        if Use._venv_is_win():
            return venv_root / Use._venv_windows_path()
        else:
            return venv_root / Use._venv_unix_path()

    @staticmethod
    def _find_entry_point(package, version):
        pkg_path = Use._venv_pkg_path(package, version)
        rec_path = (pkg_path
            / f'{package}-{version}.dist-info'
            / 'RECORD')
        contents = list(
          map(
            operator.itemgetter(0),
            map(
              functools.partialmethod(
                str.partition, ',')._make_unbound_method(),
              rec_path.read_text(
                  encoding='UTF-8'
              ).strip().splitlines()
            )
          )
        )
        contents_abs = list(map(pkg_path.__truediv__, contents))
        pkg_prefix = next(iter([
          c for c in contents_abs if c.name == 'top_level.txt'
        ])).read_text().strip()
        if pkg_prefix != package:
            entry_suffixes = (
              Path(pkg_prefix)/package/'__init__.py',
              Path(pkg_prefix)/f'{package}.py',
            )
        else:
            entry_suffixes = (
              Path(package)/'__init__.py',
              Path(f'{package}.py'),
            )
        entry_path = next(iter(
          filter(None, map(
            lambda c: c if any(str(c).rfind(str(e)) != -1 \
                            for e in entry_suffixes) else None,
            contents_abs
          ))
        ))
        return (pkg_prefix, entry_path)

    @staticmethod
    def _clean_sys_modules(package):
        del_mods = dict([(k, v.__spec__.loader) for k,v in sys.modules.items() if getattr(v, "__spec__", None) and isinstance(v.__spec__.loader, SourceFileLoader) and (k.startswith(f"{package}.") or k == package)])
        for k in del_mods:
            del sys.modules[k]

    @staticmethod
    def _venv_root(package, version):
        venv_root = Path.home() / ".justuse-python" / "venv" / package / version
        if not venv_root.exists():
            venv_root.mkdir(parents=True)
        return venv_root

    @staticmethod
    def _venv_is_win():
        return sys.platform.lower().startswith("win")

    @staticmethod
    def _venv_unix_path():
        ver = ".".join(map(str, sys.version_info[0:2]))
        return Path("lib") / f"python{ver}" / "site-packages"

    @staticmethod
    def _venv_windows_path():
        return Path("Lib") / "site-packages"

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
            "(?P<ext>whl|zip|egg)",
            filename,
        )
        return match.groupdict() if match else {}

    @staticmethod
    def _is_version_satisfied(info, sys_version):
        """
        @see https://warehouse.readthedocs.io/api-reference/json.html
        @see https://packaging.pypa.io/en/latest/specifiers.html
        """
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
        """
        Filename as API, seriously WTF...
        """
        assert isinstance(info, dict)
        assert isinstance(platform_tags, frozenset)
        if "platform_tag" not in info:
            info.update(Use._parse_filename(info["filename"]))
        our_python_tag = "".join(
            (packaging.tags.interpreter_name(), packaging.tags.interpreter_version())
        )
        python_tag = info.get("python_tag", "")
        platform_tag = info.get("platform_tag", "")
        platform_srs = {*map(str, platform_tags)}
        log.debug(f"_is_platform_compatible: {platform_srs=!r}")
        for one_platform_tag in platform_tag.split("."):
            if one_platform_tag in platform_srs and our_python_tag == python_tag:
                log.debug(
                    f"_is_platform_compatible:"
                    f"  - %30r in platform_srs?  %s",
                    one_platform_tag,
                    platform_srs,
                    one_platform_tag in platform_srs
                )
                return True
        return include_sdist and any(
            info["filename"].endswith(ext) for ext in ("egg", "zip")
        ) and not "py2" in info["filename"]

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
        return_info=False,
    ) -> Union[VerHash|Dict[str,dict]]:
        """Pick from a list of possible urls and return the
        `(version, hash_value)`
        pair from the best-fitting artifact,
        preferring bdist over sdist distribution archives."""
        info_s = sorted(urls, key=lambda i: i.get("packagetype", ""))
        for include_sdist in (False, True):  # prefer non-source
            results = [
                (
                 VerHash(info["version"], info["digests"][hash_algo])
                 if not return_info
                 else info
                )
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
        if not sys_version:
            sys_version = Version(".".join(map(str, sys.version_info[0:3])))
        if not platform_tags:
            platform_tags = get_supported()
        if "platform_tag" not in info:
            info.update(Use._parse_filename(info["filename"]))
        assert isinstance(sys_version, Version)
        return (
            (include_sdist and info["filename"][-4:] in (".zip",))
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
        return_info=False,
    ) -> Union[VerHash|Dict[str,dict]]:
        assert isinstance(releases, dict)
        assert isinstance(hash_algo, str)
        # update the release dicts to hold all info canonically
        # be aware, the json returned from pypi ["releases"] can
        # contain empty lists as dists :/
        for ver, infos in reversed(releases.items()):
            for info in infos:
                info["version"] = ver
            result = use._find_matching_artifact(
                infos,
                hash_algo=hash_algo,
                sys_version=sys_version,
                platform_tags=platform_tags,
                return_info=return_info
            )
            if result:
                return result

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

    @staticmethod
    def _ensure_proxy(mod):
        if mod.__class__ is not ModuleType:
            return mod
        return ProxyModule(mod)

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
        log.debug(f"use-url: {url}")
        exc = None
        assert url is not None, f"called with url == {url!r}"
        assert url != "None", f"called with url == {url!r}"

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
        return self._ensure_proxy(mod)

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
        """Import a module from a path.

        https://github.com/amogorkon/justuse/wiki/Use-Path

        Args:
            path ([type]): must be a pathlib.Path
            initial_globals ([type], optional): Dict that should be globally available to the module before executing it. Defaults to None.
            default ([type], optional): Return instead if an exception is encountered.
            aspectize ([type], optional): Aspectize callables. Defaults to None.
            modes (int, optional): [description]. Defaults to 0.

        Returns:
            Optional[ModuleType]: The module if it was imported, otherwise whatever was specified as default.
        """
        log.debug(f"use-path: {path}")
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
            if source_dir is None:
                source_dir = Path(main_mod.__loader__.path).parent
            if not source_dir.joinpath(path).exists():
                source_dir = Path.cwd()
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
                    del self._using[name]
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
        return self._ensure_proxy(mod)

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
            return self._ensure_proxy(mod)
        except:
            exc = traceback.format_exc()
            return Use._fail_or_default(default, ImportError, exc)

    @staticmethod
    def _get_version(name=None, package_name=None, /, mod=None) -> Optional[Version]:
        assert name is None or isinstance(name, str)
        version: Optional[Union[Callable[...],Version]|Version|str] = None
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
            version = version.__call__()
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
        fatal_exceptions |= "ERRORS" in os.environ
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
                return self._ensure_proxy(mod)
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
        return self._ensure_proxy(mod)
        
    @staticmethod
    def _get_package_data(package_name):
        response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
        data = response.json()
        for v, infos in data["releases"].items():
            for info in infos:
                info["version"] = v
                info.update(
                    Use._parse_filename(info["filename"]) or {}
                )
        return data

    @staticmethod
    def _get_filtered_data(data):
        filtered = { "urls": [], "releases": {} }
        for ver, infos in data["releases"].items():
            filtered["releases"][ver] = []
            for info in infos:
                info["version"] = ver
                info.update(
                    Use._parse_filename(info["filename"]) or {}
                )
                if not Use._is_compatible(
                    info,
                    hash_algo=Use.Hash.sha256.name,
                    sys_version=None,
                    platform_tags=None,
                    include_sdist=False
                ):
                    continue
                filtered["urls"].append(info)
                filtered["releases"][ver].append(info)
        if not filtered["urls"]:
            for ver, infos in data["releases"].items():
                for info in infos:
                    if (info["packagetype"] == "sdist" and
                        info["python_version"] == "source"
                    ):
                        filtered["urls"].append(info)
                        filtered["releases"][ver].append(info)
        for ver in data["releases"].keys():
            if not filtered["releases"][ver]:
                del filtered["releases"][ver]
        return filtered

    @__call__.register(str)
    def _use_str(
        self,
        name: str,
        /,
        *,
        path=None,
        url=None,
        version: str = None,
        hash_algo=Hash.sha256,
        hashes: Optional[Union[str, list[str]]] = None,
        default=mode.fastfail,
        aspectize=None,
        modes: int = 0,
        fatal_exceptions: bool = False,
        package_name: str = None,  # internal use
        module_name: str = None,  # internal use
    ) -> Optional[ModuleType]:
        """
        Import a package by name.

        https://github.com/amogorkon/justuse/wiki/Use-String

        Args:
            name (str): The name of the package to import.
            version (str or Version, optional): The version of the package to import. Defaults to None.
            hash_algo (member of Use.Hash, optional): For future compatibility with more modern hashing algorithms. Defaults to Hash.sha256.
            hashes (str | [str]), optional): A single hash or list of hashes of the package to import. Defaults to None.
            default (anything, optional): Whatever should be returned in case there's a problem with the import. Defaults to mode.fastfail.
            aspectize (dict, optional): Aspectize callables. Defaults to None.
            modes (int, optional): Any combination of Use.modes . Defaults to 0.
            fatal_exceptions (bool, optional): All exceptions are fatal. Defaults to True.
            package_name (str, optional): internal use only. Defaults to None.
            module_name (str, optional): internal use only. Defaults to None.

        Raises:
            RuntimeWarning: May be raised if the auto-installation of the package fails for some reason.

        Returns:
            Optional[ModuleType]: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-str: {name}")
        self.modes = modes
        aspectize = aspectize or {}

        if not hashes:
            hashes = set()
        if isinstance(hashes, str):
            hashes = set([hashes])

        # we use boolean flags to reduce the complexity of the call signature
        fatal_exceptions = bool(Use.fatal_exceptions & modes) or "ERRORS" in os.environ
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
        all_that_hash = set()
        if name in self._using:
            spec = self._using[name].spec
        elif not auto_install:
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
                return self._ensure_proxy(mod)
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
                return self._ensure_proxy(mod)
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
            if target_version and not hashes:  # let's try to be helpful
                response = requests.get(
                    f"https://pypi.org/pypi/{package_name}/{target_version}/json"
                )
                if response.status_code == 404:
                    raise RuntimeWarning(
                        f"Are you sure {package_name} with version {version} exists?"
                    )
                elif response.status_code != 200:
                    raise RuntimeWarning(
                        f"Something bad happened while contacting PyPI for info on {package_name} ( {response.status_code} ), which we tried to look up because a matching hashes for the auto-installation was missing."
                    )
                data = response.json()

                version, that_hash = hit = Use._find_matching_artifact(
                    data["urls"], hash_algo=hash_algo.name
                )
                log.info(f"{hit=} from  Use._find_matching_artifact")
                if that_hash:
                    if that_hash is not None:
                        hashes.add(that_hash)

                    raise RuntimeWarning(
                        f"""Failed to auto-install '{package_name}' because hash_value is missing. This may work:
use("{package_name}", version="{version}", hashes={hashes!r}, modes=use.auto_install)"""
                    )
                raise RuntimeWarning(
                    f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
                )
            elif not target_version and hashes:
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

                if not target_version and (hash_value or hashes):
                    hashes.add(hash_value)
                    raise RuntimeWarning(
                        f"""Please specify version and hash for auto-installation of '{package_name}'.
To get some valuable insight on the health of this package, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version: use("{name}", version="{version}", hashes={hashes!r}, modes=use.auto_install)
"""
                    )

            # if it's a pure python package, there is only an artifact, no installation
            query = self.registry.execute(
                "SELECT id, installation_path FROM distributions WHERE name=? AND version=?",
                (name, version),
            ).fetchone()

            if query:
                query = self.registry.execute(
                    "SELECT path FROM artifacts WHERE distribution_id=?",
                    [
                        query["id"],
                    ],
                ).fetchone()
            if query:
                path = Path(query["path"])
                if not path.exists():
                    path = None
            if path and not url:
                url = URL(f"file:/{path.absolute()}")
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
                    if "urls" not in data:
                        return Use._fail_or_default(
                            default,
                            Use.AutoInstallationError,
                            f"Tried to auto-install {package_name} {target_version} but failed because no valid URLs to download could be found.",
                        )
                    for entry in data["urls"]:
                        if not entry["url"] or ":" not in entry["url"]:
                            continue
                        url = URL(entry["url"])
                        path = (
                            self.home
                            / "packages"
                            / Path(url.asdict()["path"]["segments"][-1]).name
                        )
                        log.error("url = %s", url)
                        entry["version"] = str(target_version)
                        log.debug(f"looking at {entry=}")
                        assert isinstance(all_that_hash, set)
                        all_that_hash.add(that_hash := entry["digests"].get(hash_algo.name))
                        if hashes.intersection(all_that_hash):
                            found = (entry, that_hash)
                            hit = VerHash(version, that_hash)
                            log.info(f"Matches user hash: {entry=} {hit=}")
                            break
                    if found is None:
                        return Use._fail_or_default(
                            default,
                            Use.AutoInstallationError,
                            f"Tried to auto-install {name!r} ({package_name=!r}) with {target_version=!r} but failed because none of the available hashes ({all_that_hash=!r}) match the expected hash ({hash_value=!r} or {hashes=!r}).",
                        )
                    entry, that_hash = found
                    hash_value = that_hash
                    if that_hash is not None:
                        assert isinstance(hashes, set)
                        hashes.add(that_hash)
                except KeyError as be:  # json issues
                    msg = f"request to https://pypi.org/pypi/{package_name}/{target_version}/json lead to an error: {be}"
                    raise RuntimeError(msg, response) from be
                if exc:
                    return Use._fail_or_default(
                        default,
                        Use.AutoInstallationError,
                        f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI.",
                    )
                # we've got a complete JSON with a matching entry, let's download
                if not path:
                    path = (
                        self.home
                        / "packages"
                        / Path(url.asdict()["path"]["segments"][-1]).name
                    )

                if not path.exists():
                    print("Downloading", url, "...")
                    download_response = requests.get(str(url), allow_redirects=True)
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

            if name in self._hacks:
                # if version in self._hacks[name]:
                mod = self._hacks[name](
                    package_name=package_name,
                    version=Use._get_version(mod=mod),
                    url=url,
                    path=path,
                    that_hash=hash_value,
                    folder=folder,
                    fatal_exceptions=fatal_exceptions,
                    module_name=module_name,
                )
                return self._ensure_proxy(mod)

            # trying to import directly from zip
            try:
                importer = zipimporter(path)
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
        assert mod, f"Well. Shit. No module. ( {path} )"
        this_version = Use._get_version(mod=mod) or version
        assert this_version, f"Well. Shit, no version. ( {path} )"
        self._save_module_info(
            name=name,
            version=this_version,
            artifact_path=path,
            hash_value=that_hash,
            installation_path=folder,
        )
        return self._ensure_proxy(mod)

    pass


# we should avoid side-effects during testing, specifically for the version-upgrade-warning

# ALIASES
Use.Version = Version
Use.config = config
Use.mode = mode
Use.Path = Path
Use.URL = URL
Use.__path__ = str(Path(__file__).resolve().parent)
Use.__name__ = __name__


use = Use()
if not test_version:
    sys.modules["use"] = use

# no circular import this way
hacks_path = Path(__file__).parent / "package_hacks.py"
assert hacks_path.exists()
use(hacks_path)  # type: ignore