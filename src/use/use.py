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
- support P2P pkg distribution (TODO)
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
>>> tools = use(use._ensure_path("/media/sf_Dropbox/code/tools.py"), reloading=True)

# it is possible to import standalone modules from online sources
# with immediate sha1-hash-verificiation before execution of the code like
>>> utils = use(use.URL("https://raw.githubusercontent.com/PIA-Group/BioSPPy/7696d682dc3aafc898cd9161f946ea87db4fed7f/biosppy/utils.py"),
                    hashes="95f98f25ef8cfa0102642ea5babbe6dde3e3a19d411db9164af53a9b4cdcccd8")

# to auto-install a certain version (within a virtual env and pip in secure hash-check mode) of a pkg you can do
>>> np = use("numpy", version="1.1.1", modes=use.auto_install, hash_value=["9879de676"])

:author: Anselm Kiefner (amogorkon)
:author: David Reilly
:license: MIT
"""

#% Preamble
# we use https://github.com/microsoft/vscode-python/issues/17218 with % syntax to structure the code

# Read in this order:
# 1) initialization (instance of Use() is set as module on import)
# 2) use() dispatches to one of three __call__ methods, depending on first argument
# 3) from there, various global functions are called
# 4) a ProxyModule is always returned, wrapping the module that was imported


#% Imports

from __future__ import annotations

import ast
import asyncio
import atexit
import hashlib
import importlib.util
import inspect
import os
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from enum import Enum
from functools import lru_cache as cache
from functools import partial, partialmethod, reduce, singledispatch, update_wrapper
from importlib import metadata
from importlib.abc import Finder, Loader
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from importlib.util import find_spec
from inspect import isfunction, ismethod  # for aspectizing, DO NOT REMOVE
from itertools import chain, takewhile
from logging import DEBUG, INFO, NOTSET, WARN, StreamHandler, getLogger, root
from pathlib import Path, PureWindowsPath, WindowsPath
from subprocess import PIPE, run
from textwrap import dedent
from types import FrameType, ModuleType
from typing import Any, Callable, Optional, Union
from warnings import warn

import requests
import toml
from beartype import beartype
from furl import furl as URL
from icontract import ensure, invariant, require
from packaging import tags
from packaging.specifiers import SpecifierSet
from pip._internal.utils import compatibility_tags
from .modules import Decorators as D
from .modules.Hashish import Hash
from .modules.Mod import ProxyModule, ModuleReloader
from .modules.Messages import (
    AmbiguityWarning,
    Message,
    NoValidationWarning,
    NotReloadableWarning,
    UnexpectedHash,
    VersionWarning,
)
from .modules.init_conf import (
    config,
    log,
    mode,
    ModInUse,
    test_config,
    test_version,
    _reloaders,
    _using,
    __version__,
)
from .modules.install_utils import (
    _apply_aspect,
    _auto_install,
    _build_mod,
    _ensure_path,
    _ensure_version,
    _fail_or_default,
    _import_public_no_install,
    _parse_name,
    _pebkac_version_no_hash,
    _pebkac_no_version_hash,
    _pebkac_no_version_no_hash,
)

from use import hash_alphabet, pypi_model

os.chdir(Path(__file__).parent)

use = sys.modules.get(__name__)
from pypi_model import PyPI_Project, PyPI_Release, Version


# sometimes all you need is a sledge hammer..
_orig_locks = threading._shutdown_locks


def atexit_hook():
    global _reloaders
    for lock in threading._shutdown_locks:
        lock.unlock()
    for reloader in _reloaders.values():
        reloader.stop()
    for lock in threading._shutdown_locks:
        lock.unlock()


from atexit import register

atexit.register(atexit_hook)


#%% Version and Packaging


class PlatformTag:
    def __init__(self, platform: str):
        self.platform = platform

    def __str__(self):
        return self.platform

    def __repr__(self):
        return f"use.PlatformTag({self.platform!r})"

    def __hash__(self):
        return hash(self.platform)

    @require(lambda self, other: isinstance(other, self.__class__))
    def __eq__(self, other):
        return self.platform == other.platform


#% Main Classes


class Info(dict):
    def __repr__(self):
        return "<Info of size %d>" % len(self)


class Use(ModuleType):
    # MODES to reduce signature complexity
    # enum.Flag wasn't viable, but this alternative is actually pretty cool
    auto_install = 2 ** 0
    fatal_exceptions = 2 ** 1
    reloading = 2 ** 2
    no_public_installation = 2 ** 4

    def __init__(self):
        # TODO for some reason removing self._using isn't as straight forward..
        self._using = _using
        self.home: Path

        self._set_up_files_and_directories()
        # might run into issues during testing otherwise
        self.registry = self._set_up_registry()
        self._user_registry = toml.load(self.home / "user_registry.toml")

        # for the user to copy&paste
        with open(self.home / "default_config.toml", "w") as file:
            toml.dump(config, file)

        with open(self.home / "config.toml") as file:
            config.update(toml.load(file))

        config.update(test_config)

        if config["debugging"]:
            root.setLevel(DEBUG)

        if config["version_warning"]:
            try:
                response = requests.get("https://pypi.org/pypi/justuse/json")
                data = response.json()
                max_version = max(
                    Version(version) for version in data["releases"].keys()
                )
                target_version = max_version
                this_version = __version__
                if Version(this_version) < target_version:
                    warn(
                        Message.use_version_warning(target_version),
                        VersionWarning,
                    )
            except (KeyError, requests.exceptions.ConnectionError):
                if test_version:
                    raise
                log.error(
                    traceback.format_exc()
                )  # we really don't need to bug the user about this (either pypi is down or internet is broken)

    def _set_up_files_and_directories(self):
        self.home = Path(
            os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python"))
        )
        try:
            self.home.mkdir(mode=0o755, parents=True, exist_ok=True)
        except PermissionError:
            # this should fix the permission issues on android #80
            self.home = _ensure_path(tempfile.mkdtemp(prefix="justuse_"))
        (self.home / "packages").mkdir(mode=0o755, parents=True, exist_ok=True)
        for file in (
            "config.toml",
            "config_defaults.toml",
            "usage.log",
            "registry.db",
            "user_registry.toml",
        ):
            (self.home / file).touch(mode=0o755, exist_ok=True)

    def _sqlite_row_factory(self, cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    def _set_up_registry(self, path: Optional[Path] = None):
        registry = None
        if path or test_version and "DB_TEST" not in os.environ:
            registry = sqlite3.connect(path or ":memory:").cursor()
        else:
            try:
                registry = sqlite3.connect(self.home / "registry.db").cursor()
            except Exception as e:
                raise RuntimeError(Message.couldnt_connect_to_db(e))
        registry.row_factory = self._sqlite_row_factory
        registry.execute("PRAGMA foreign_keys=ON")
        registry.execute("PRAGMA auto_vacuum = FULL")
        registry.executescript(
            """
CREATE TABLE IF NOT EXISTS "artifacts" (
	"id"	INTEGER,
	"distribution_id"	INTEGER,
	"import_relpath" TEXT,
	"artifact_path"	TEXT,
  "module_path" TEXT,
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
        registry.connection.commit()
        return registry

    def recreate_registry(self):
        self.registry.close()
        self.registry.connection.close()
        self.registry = None
        number_of_backups = len(list((self.home / "registry.db").glob("*.bak")))
        (self.home / "registry.db").rename(
            self.home / f"registry.db.{number_of_backups + 1}.bak"
        )
        (self.home / "registry.db").touch(mode=0o644)
        self.registry = self._set_up_registry()
        self.cleanup()

    def execute_wrapped(self, *args, **kwargs):
        try:
            return self.registry.execute(*args, **kwargs)
        except sqlite3.OperationalError as _oe:
            self.recreate_registry()
            return self.registry.execute(*args, **kwargs)

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
            raise RuntimeWarning("__builtins__ is something unexpected")

    def del_entry(self, name, version):
        # TODO: CASCADE to artifacts etc
        self.registry.execute(
            "DELETE FROM hashes WHERE artifact_id IN (SELECT id FROM artifacts WHERE distribution_id IN (SELECT id FROM distributions WHERE name=? AND version=?))",
            (name, version),
        )
        self.registry.execute(
            "DELETE FROM artifacts WHERE distribution_id IN (SELECT id FROM distributions WHERE name=? AND version=?)",
            (name, version),
        )
        self.registry.execute(
            "DELETE FROM distributions WHERE name=? AND version=?", (name, version)
        )
        self.registry.connection.commit()

    def cleanup(self):
        """Bring registry and downloaded packages in sync.

        First all packages are removed that don't have a matching registry entry, then all registry entries that don't have a matching pkg.
        """

        def delete_folder(path):
            for sub in path.iterdir():
                if sub.is_dir():
                    delete_folder(sub)
                else:
                    sub.unlink()
            path.rmdir()

        for name, version, artifact_path, installation_path in self.registry.execute(
            "SELECT name, version, artifact_path, installation_path FROM distributions JOIN artifacts on distributions.id = distribution_id"
        ).fetchall():
            if not (
                _ensure_path(artifact_path).exists()
                and _ensure_path(installation_path).exists()
            ):
                self.del_entry(name, version)
        self.registry.connection.commit()

    def _save_module_info(
        self,
        *,
        version: Version,
        artifact_path: Optional[Path],
        hash_value=Optional[str],
        installation_path=Path,
        module_path: Optional[Path],
        name: str,
        import_relpath: str,
        hash_algo=Hash.sha256,
    ):
        """Update the registry to contain the pkg's metadata."""
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
INSERT OR IGNORE INTO artifacts (distribution_id, import_relpath, artifact_path, module_path)
VALUES ({self.registry.lastrowid}, '{import_relpath}', '{artifact_path}', '{module_path}')
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
        self._using[name] = ModInUse(name, mod, path, spec, frame)

    @D.methdispatch
    def __call__(self, thing, /, *args, **kwargs):
        raise NotImplementedError(Message.cant_use(thing))

    @require(lambda hash_algo: hash_algo in Hash)
    @require(lambda as_import: as_import.isidentifier())
    @__call__.register(URL)
    def _use_url(
        self,
        url: URL,
        /,
        *,
        hash_algo=Hash.sha256,
        hash_value=None,
        initial_globals: Optional[dict[Any, Any]] = None,
        as_import: str = None,
        default=mode.fastfail,
        modes=0,
    ) -> ProxyModule:
        log.debug(f"use-url: {url}")
        exc = None

        response = requests.get(str(url))
        if response.status_code != 200:
            raise ImportError(Message.web_error())
        this_hash = hash_algo.value(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                return _fail_or_default(
                    UnexpectedHash(
                        f"{this_hash} does not match the expected hash {hash_value} - aborting!"
                    ),
                    default,
                )
        else:
            warn(Message.no_validation(url, hash_algo, this_hash), NoValidationWarning)

        name = url.path.segments[-1]
        try:
            mod = _build_mod(
                name=name,
                code=response.content,
                module_path=_ensure_path(url.path),
                initial_globals=initial_globals,
            )
        except KeyError:
            raise
        if exc:
            return _fail_or_default(ImportError(exc), default)

        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        if as_import:
            sys.modules[as_import] = mod
        return ProxyModule(mod)

    @require(lambda as_import: as_import.isidentifier())
    @__call__.register(Path)
    def _use_path(
        self,
        path,
        /,
        *,
        initial_globals=None,
        as_import: str = None,
        default=mode.fastfail,
        modes=0,
    ) -> ProxyModule:
        """Import a module from a path.

        https://github.com/amogorkon/justuse/wiki/Use-Path

        Args:
            path ([type]): must be a pathlib.Path
            initial_globals ([type], optional): dict that should be globally available to the module before executing it. Defaults to None.
            default ([type], optional): Return instead if an exception is encountered.
            modes (int, optional): [description]. Defaults to 0; Acceptable mode for this variant: use.reloading.

        Returns:
            Optional[ModuleType]: The module if it was imported, otherwise whatever was specified as default.
        """
        log.debug(f"use-path: {path}")
        initial_globals = initial_globals or {}

        reloading = bool(Use.reloading & modes)

        exc = None
        mod = None

        if path.is_dir():
            return _fail_or_default(
                ImportError(f"Can't import directory {path}"), default
            )

        original_cwd = source_dir = Path.cwd()
        try:
            if not path.is_absolute():
                source_dir = Path.cwd()

            # calling from another use()d module
            # let's see where we started
            main_mod = __import__("__main__")
            if source_dir and source_dir.exists():
                os.chdir(source_dir.parent)
                source_dir = source_dir.parent
            else:
                # there are a number of ways to call use() from a non-use() starting point
                # let's first check if we are running in jupyter
                jupyter = "ipykernel" in sys.modules
                # we're in jupyter, we use the CWD as set in the notebook
                if not jupyter and hasattr(main_mod, "__file__"):
                    source_dir = (
                        _ensure_path(
                            inspect.currentframe().f_back.f_back.f_code.co_filename
                        )
                        .resolve()
                        .parent
                    )
            if source_dir is None:
                if main_mod.__loader__ and hasattr(main_mod.__loader__, "path"):
                    source_dir = _ensure_path(main_mod.__loader__.path).parent
                else:
                    source_dir = Path.cwd()
            if not source_dir.joinpath(path).exists():
                if files := [*[*source_dir.rglob(f"**/{path}")]]:
                    source_dir = _ensure_path(files[0]).parent
                else:
                    source_dir = Path.cwd()
            if not source_dir.exists():
                return _fail_or_default(
                    NotImplementedError(
                        "Can't determine a relative path from a virtual file."
                    ),
                    default,
                )
            path = source_dir.joinpath(path).resolve()
            if not path.exists():
                return _fail_or_default(ImportError(f"Sure '{path}' exists?"), default)
            os.chdir(path.parent)
            name = path.stem
            if reloading:
                try:
                    with open(path, "rb") as file:
                        code = file.read()
                    # initial instance, if this doesn't work, just throw the towel
                    mod = _build_mod(
                        name=name,
                        code=code,
                        initial_globals=initial_globals,
                        module_path=path.resolve(),
                    )
                except KeyError:
                    exc = traceback.format_exc()
                if exc:
                    return _fail_or_default(ImportError(exc), default)
                mod = ProxyModule(mod)
                reloader = ModuleReloader(
                    proxy=mod,
                    name=name,
                    path=path,
                    initial_globals=initial_globals,
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
                    isfunction(value)
                    for key, value in mod.__dict__.items()
                    if key not in initial_globals.keys() and not key.startswith("__")
                ):
                    warn(Message.not_reloadable(name), NotReloadableWarning)
            else:  # NOT reloading
                with open(path, "rb") as file:
                    code = file.read()
                # the path needs to be set before attempting to load the new module - recursion confusing ftw!
                frame = inspect.getframeinfo(inspect.currentframe())
                self._set_mod(name=name, mod=mod, frame=frame)
                try:
                    mod = _build_mod(
                        name=name,
                        code=code,
                        initial_globals=initial_globals,
                        module_path=path,
                    )
                except KeyError:
                    del self._using[name]
                    exc = traceback.format_exc()
        except KeyError:
            exc = traceback.format_exc()
        finally:
            # let's not confuse the user and restore the cwd to the original in any case
            os.chdir(original_cwd)
        if exc:
            return _fail_or_default(ImportError(exc), default)
        if as_import:
            sys.modules[as_import] = mod
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return ProxyModule(mod)

    @__call__.register(
        type(None)
    )  # singledispatch is picky - can't be anything but a type
    def _use_kwargs(
        self,
        _: None,  # sic! otherwise single-dispatch with 'empty' *args won't work
        /,
        *,
        package_name: str = None,
        module_name: str = None,
        version: str = None,
        hash_algo=Hash.sha256,
        hashes: Optional[Union[str, list[str]]] = None,
        default=mode.fastfail,
        modes: int = 0,
    ) -> ProxyModule:
        """
        Import a pkg by name.

        https://github.com/amogorkon/justuse/wiki/Use-String

        Args:
            name (str): The name of the pkg to import.
            version (str or Version, optional): The version of the pkg to import. Defaults to None.
            hash_algo (member of Use.Hash, optional): For future compatibility with more modern hashing algorithms. Defaults to Hash.sha256.
            hashes (str | [str]), optional): A single hash or list of hashes of the pkg to import. Defaults to None.
            default (anything, optional): Whatever should be returned in case there's a problem with the import. Defaults to mode.fastfail.
            modes (int, optional): Any combination of Use.modes . Defaults to 0.

        Raises:
            RuntimeWarning: May be raised if the auto-installation of the pkg fails for some reason.

        Returns:
            Optional[ModuleType]: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-kwargs: {package_name} {module_name} {version} {hashes}")
        return self._use_package(
            name=f"{package_name}/{module_name}",
            package_name=package_name,
            module_name=module_name,
            version=version,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
        )

    @__call__.register(tuple)
    def _use_tuple(
        self,
        pkg_tuple,
        /,
        *,
        version: str = None,
        hash_algo=Hash.sha256,
        hashes: Optional[Union[str, list[str]]] = None,
        default=mode.fastfail,
        modes: int = 0,
    ) -> ProxyModule:
        """
        Import a pkg by name.

        https://github.com/amogorkon/justuse/wiki/Use-String

        Args:
            name (str): The name of the pkg to import.
            version (str or Version, optional): The version of the pkg to import. Defaults to None.
            hash_algo (member of Use.Hash, optional): For future compatibility with more modern hashing algorithms. Defaults to Hash.sha256.
            hashes (str | [str]), optional): A single hash or list of hashes of the pkg to import. Defaults to None.
            default (anything, optional): Whatever should be returned in case there's a problem with the import. Defaults to mode.fastfail.
            modes (int, optional): Any combination of Use.modes . Defaults to 0.

        Raises:
            RuntimeWarning: May be raised if the auto-installation of the pkg fails for some reason.

        Returns:
            Optional[ModuleType]: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-tuple: {pkg_tuple} {version} {hashes}")
        package_name, module_name = pkg_tuple
        return self._use_package(
            name=f"{package_name}/{module_name}",
            package_name=package_name,
            module_name=module_name,
            version=version,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
        )

    @__call__.register(str)
    def _use_str(
        self,
        name: str,
        /,
        *,
        version: str = None,
        hash_algo=Hash.sha256,
        hashes: Optional[Union[str, list[str]]] = None,
        default=mode.fastfail,
        modes: int = 0,
    ) -> ProxyModule:
        """
        Import a pkg by name.

        https://github.com/amogorkon/justuse/wiki/Use-String

        Args:
            name (str): The name of the pkg to import.
            version (str or Version, optional): The version of the pkg to import. Defaults to None.
            hash_algo (member of Use.Hash, optional): For future compatibility with more modern hashing algorithms. Defaults to Hash.sha256.
            hashes (str | [str]), optional): A single hash or list of hashes of the pkg to import. Defaults to None.
            default (anything, optional): Whatever should be returned in case there's a problem with the import. Defaults to mode.fastfail.
            modes (int, optional): Any combination of Use.modes . Defaults to 0.

        Raises:
            RuntimeWarning: May be raised if the auto-installation of the pkg fails for some reason.

        Returns:
            Optional[ModuleType]: Module if successful, default as specified otherwise.
        """
        package_name, module_name = _parse_name(name)
        return self._use_package(
            name=name,
            package_name=package_name,
            module_name=module_name,
            version=version,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
        )

    def _use_package(
        self,
        *,
        name,
        package_name,
        module_name,
        version,
        hashes,
        modes,
        default,
        hash_algo,
        user_msg=Message,
    ):
        log.debug(
            f"use-package: {name}, {package_name}, {module_name}, {version}, {hashes}"
        )
        if isinstance(hashes, str):
            hashes = set([hashes])
        if not hashes:
            hashes = set()
        hashes = {
            H
            if len(H) == 64
            else hash_alphabet.num_as_hexdigest(hash_alphabet.JACK_as_num(H))

            for H in hashes
        }
        rest = module_name

        # we use boolean flags to reduce the complexity of the call signature
        auto_install = bool(Use.auto_install & modes)

        version: Version = Version(version) if version else None

        # The "try and guess" behaviour is due to how classical imports work,
        # which is inherently ambiguous, but can't really be avoided for packages.
        # let's first see if the user might mean something else entirely
        if _ensure_path(f"./{module_name}.py").exists():
            warn(Message.ambiguous_name_warning(name), AmbiguityWarning)
        spec = None

        if name in self._using:
            spec = self._using[name].spec
        elif not auto_install:
            spec = importlib.util.find_spec(package_name)

        # welcome to the buffet table, where everything is a lie
        # fmt: off
        case = (bool(version), bool(hashes), bool(spec), bool(auto_install))
        log.info("case = %s", case)
        case_func = {
            (0, 0, 0, 0): lambda **kwargs: ImportError(Message.cant_import(name)),
            (0, 0, 0, 1): _pebkac_no_version_no_hash,
            (0, 0, 1, 0): _import_public_no_install,
            (0, 1, 0, 0): lambda **kwargs: ImportError(Message.cant_import(name)),
            (1, 0, 0, 0): lambda **kwargs: ImportError(Message.cant_import(name)),
            (0, 0, 1, 1): lambda **kwargs: _auto_install(_import_public_no_install, **kwargs),
            (0, 1, 1, 0): _import_public_no_install,
            (1, 1, 0, 0): lambda **kwargs: ImportError(Message.cant_import(name)),
            (1, 0, 0, 1): _pebkac_version_no_hash,
            (1, 0, 1, 0): lambda **kwargs: _ensure_version(_import_public_no_install, **kwargs),
            (0, 1, 0, 1): _pebkac_no_version_hash,
            (0, 1, 1, 1): lambda **kwargs: _pebkac_no_version_hash(_import_public_no_install, **kwargs),
            (1, 0, 1, 1): lambda **kwargs: _pebkac_version_no_hash(_ensure_version(_import_public_no_install, **kwargs), **kwargs),
            (1, 1, 0, 1): _auto_install,
            (1, 1, 1, 0): lambda **kwargs: _ensure_version(_import_public_no_install, **kwargs),
            (1, 1, 1, 1): lambda **kwargs: _auto_install(_ensure_version(_import_public_no_install, **kwargs), **kwargs),
        }[case]
        result = case_func(**locals())
        log.info("result = %s", result)
        # fmt: on
        assert result

        if isinstance((mod := result), ModuleType):
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=mod, spec=spec, frame=frame)
            return ProxyModule(mod)
        return _fail_or_default(result, default)


use = Use()
use.__dict__.update(
    {k: v for k, v in globals().items()}
)  # to avoid recursion-confusion
use = ProxyModule(use)


def decorator_log_calling_function_and_args(func, *args):
    """
    Decorator to log the calling function and its arguments.

    Args:
        func (function): The function to decorate.
        *args: The arguments to pass to the function.

    Returns:
        function: The decorated function.
    """

    def wrapper(*args, **kwargs):
        log.debug(f"{func.__name__}({args}, {kwargs})")
        return func(*args, **kwargs)

    return wrapper


use @ (isfunction, "", beartype)
use @ (isfunction, "", decorator_log_calling_function_and_args)

if not test_version:
    sys.modules["use"] = use
