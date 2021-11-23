"""
Main classes that act as API for the user to interact with.
"""


import asyncio
import atexit
import hashlib
import importlib.util
import inspect
import os
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from atexit import register
from logging import DEBUG, INFO, NOTSET, getLogger, root
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional, Type, Union
from warnings import warn

import requests
import toml
from furl import furl as URL
from icontract import ensure, invariant, require

from use import (
    AmbiguityWarning,
    Hash,
    Modes,
    ModInUse,
    NotReloadableWarning,
    NoValidationWarning,
    UnexpectedHash,
    VersionWarning,
    __version__,
    buffet_table,
    config,
    home,
    isfunction,
)
from use.hash_alphabet import JACK_as_num, is_JACK, num_as_hexdigest
from use.messages import Message
from use.pimp import _apply_aspect, _build_mod, _ensure_path, _fail_or_default, _parse_name
from use.pypi_model import PyPI_Project, PyPI_Release, Version
from use.tools import methdispatch

log = getLogger(__name__)

# internal subpackage imports
test_config: str = locals().get("test_config", {})
test_version: str = locals().get("test_version", None)

_reloaders: dict["ProxyModule", "ModuleReloader"] = {}  # ProxyModule:Reloader
_using = {}

# sometimes all you need is a sledge hammer..
@register
def _release_locks():
    for _ in range(2):
        [lock.unlock() for lock in threading._shutdown_locks]
        [reloader.stop() for reloader in _reloaders.values()]


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

    def __matmul__(self, other: tuple):
        thing = self.__implementation
        check, pattern, decorator = other
        return _apply_aspect(thing, check, pattern, decorator, aspectize_dunders=False)

    def __call__(self, *args, **kwargs):
        with self.__condition:
            return self.__implementation(*args, **kwargs)


class ModuleReloader:
    def __init__(self, *, proxy, name, path, initial_globals):
        self.proxy = proxy
        "ProxyModula that we refer to."
        self.name = name
        self.path = path
        self.initial_globals = initial_globals
        self._condition = threading.RLock()
        self._stopped = True
        self._thread = None

    def start_async(self):
        loop = asyncio.get_running_loop()
        loop.create_task(self.run_async())

    @require(lambda self: self._thread is None or self._thread.is_alive())
    def start_threaded(self):
        self._stopped = False
        atexit.register(self.stop)
        self._thread = threading.Thread(target=self.run_threaded, name=f"reloader__{self.name}")
        self._thread.start()

    async def run_async(self):
        last_filehash = None
        while not self._stopped:
            with open(self.path, "rb") as file:
                code = file.read()
            current_filehash = hashlib.blake2b(code).hexdigest()
            if current_filehash != last_filehash:
                try:
                    mod = _build_mod(
                        name=self.name,
                        code=code,
                        initial_globals=self.initial_globals,
                        module_path=self.path.resolve(),
                    )
                    self.proxy.__implementation = mod
                except KeyError:
                    print(traceback.format_exc())
            last_filehash = current_filehash
            await asyncio.sleep(1)

    def run_threaded(self):
        last_filehash = None
        while not self._stopped:
            with self._condition:
                with open(self.path, "rb") as file:
                    code = file.read()
                current_filehash = hashlib.blake2b(code).hexdigest()
                if current_filehash != last_filehash:
                    try:
                        mod = _build_mod(
                            name=self.name,
                            code=code,
                            initial_globals=self.initial_globals,
                            module_path=self.path,
                        )
                        self.proxy._ProxyModule__implementation = mod
                    except KeyError:
                        print(traceback.format_exc())
                last_filehash = current_filehash
            time.sleep(1)

    def stop(self):
        self._stopped = True

    def __del__(self):
        self.stop()


class Use(ModuleType):
    # MODES to reduce signature complexity
    # enum.Flag wasn't viable, but this alternative is actually pretty cool
    auto_install = Modes.auto_install
    fatal_exceptions = Modes.fatal_exceptions
    reloading = Modes.reloading
    no_public_installation = Modes.no_public_installation

    def __init__(self):
        # TODO for some reason removing self._using isn't as straight forward..
        self._using = _using
        # might run into issues during testing otherwise
        self.registry = self._set_up_registry()
        "Registry sqlite DB to store all relevant package metadata."
        self._user_registry = toml.load(home / "user_registry.toml")
        "User can override package directories via user_registry.toml"

        # for the user to copy&paste
        with open(home / "config_defaults.toml", "w") as rfile:
            toml.dump(config, rfile)

        with open(home / "config.toml") as rfile:
            config.update(toml.load(rfile))

        config.update(test_config)

        if config["debugging"]:
            root.setLevel(DEBUG)

        if config["version_warning"]:
            try:
                response = requests.get("https://pypi.org/pypi/justuse/json")
                "Checking if there's a new version of justuse."
                data = response.json()
                max_version = max(Version(version) for version in data["releases"].keys())
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

    def _sqlite_row_factory(self, cursor, row):
        return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

    def _set_up_registry(self, path: Optional[Path] = None):
        registry = None
        if path or test_version and "DB_TEST" not in os.environ:
            registry = sqlite3.connect(path or ":memory:").cursor()
        else:
            try:
                registry = sqlite3.connect(home / "registry.db").cursor()
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
        number_of_backups = len(list((home / "registry.db").glob("*.bak")))
        (home / "registry.db").rename(home / f"registry.db.{number_of_backups + 1}.bak")
        (home / "registry.db").touch(mode=0o644)
        self.registry = self._set_up_registry()
        self.cleanup()

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
        self.registry.execute("DELETE FROM distributions WHERE name=? AND version=?", (name, version))
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
            if not (_ensure_path(artifact_path).exists() and _ensure_path(installation_path).exists()):
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

    @methdispatch
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
        default=Modes.fastfail,
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
                    UnexpectedHash(f"{this_hash} does not match the expected hash {hash_value} - aborting!"),
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
        default=Modes.fastfail,
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
            return _fail_or_default(ImportError(f"Can't import directory {path}"), default)

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
                        _ensure_path(inspect.currentframe().f_back.f_back.f_code.co_filename).resolve().parent
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
                    NotImplementedError("Can't determine a relative path from a virtual file."),
                    default,
                )
            path = source_dir.joinpath(path).resolve()
            if not path.exists():
                return _fail_or_default(ImportError(f"Sure '{path}' exists?"), default)
            os.chdir(path.parent)
            name = path.stem
            if reloading:
                try:
                    with open(path, "rb") as rfile:
                        code = rfile.read()
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
                with open(path, "rb") as rfile:
                    code = rfile.read()
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

    @__call__.register(type(None))  # singledispatch is picky - can't be anything but a type
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
        default=Modes.fastfail,
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
        default=Modes.fastfail,
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
        default=Modes.fastfail,
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
            default (anything, optional): Whatever should be returned in case there's a problem with the import. Defaults to Modes.fastfail.
            modes (int, optional): Any combination of Use.modes . Defaults to 0.

        Raises:
            RuntimeWarning: May be raised if something non-critical happens during import.
            ImportError: May be raised if the auto-installation of the pkg fails for some reason.

        Returns:
            ProxyModule|Any: Module (wrapped in a ProxyModule) if successful, default as specified if the requested Module couldn't be imported for some reason.
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
        assert hash_algo != None, "Hash algorithm must be specified"

        if isinstance(hashes, str):
            hashes = set(hashes.split())
        if not hashes:
            hashes = set()
        hashes: set = {num_as_hexdigest(JACK_as_num(H)) if is_JACK(H) else H for H in hashes}
        # we use boolean flags to reduce the complexity of the call signature
        auto_install: bool = Modes.auto_install & modes

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

        case = (bool(version), bool(hashes), bool(spec), bool(auto_install))
        log.info("case = %s", case)
        # welcome to the buffet table, where everything is a lie
        kwargs = {
            "name": name,
            "package_name": package_name,
            "module_name": module_name,
            "version": version,
            "hashes": hashes,
            "hash_algo": hash_algo,
            "user_msg": user_msg,
            "spec": spec,
            "fastfail": Modes.fastfail & modes,
            "no_public_installation": Modes.no_public_installation & modes,
            "fatal_exceptions": Modes.fatal_exceptions & modes,
            "sys_version": Version(".".join(map(str, sys.version_info[0:3]))),
        }
        result = buffet_table(case, kwargs)
        assert result

        if isinstance(result, Exception):
            return _fail_or_default(result, default)

        if isinstance(result, ModuleType):
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=result, spec=spec, frame=frame)
            return ProxyModule(result)
        raise RuntimeError(f"{result=!r} is neither a module nor an Exception")
