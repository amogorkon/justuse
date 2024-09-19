"""
Main classes that act as API for the user to interact with.
"""

import asyncio
import atexit
import contextlib
import importlib
import importlib.metadata
import inspect
import os
import py_compile
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import singledispatchmethod
from logging import DEBUG, getLogger, root
from pathlib import Path
from types import ModuleType
from typing import Any
from warnings import warn

import requests
from furl import furl as URL
from icontract import require

from use import (
    Hash,
    Modes,
    NotReloadableWarning,
    NoValidationWarning,
    UnexpectedHash,
    VersionWarning,
    __version__,
    buffet_table,
    config,
    home,
    sessionID,
)
from use.classes import ProxyModule, ModuleReloader
from use.aspectizing import _applied_decorators
from use.hash_alphabet import JACK_as_num, is_JACK
from use.messages import KwargMessage, StrMessage, TupleMessage, UserMessage
from use.pimp import (
    _build_mod,
    _ensure_path,
    _fail_or_default,
    _is_builtin,
    _parse_name,
    _real_path,
    module_from_pyc,
    _import_as,
)
from use.pydantics import Version, git

now = time.perf_counter_ns()
counter_ = 0


def timer():
    global now, counter_
    counter_ += 1
    cf = inspect.currentframe()
    print(
        f"Time to #{counter_} at L{cf.f_back.f_lineno}({Path(inspect.getframeinfo(cf).filename).name}) in {(time.perf_counter_ns() - now) / 1_000_000_000:.2f} s"
    )
    now = time.perf_counter_ns()


timer()


def excel_style_datetime(now: datetime) -> float:
    """
    Build a float representing the current time in the excel format.
    First 4 digits are the year, the next two are the month, the next two are the day followed
    by a decimal point, then time in fraction of the day.

    Args:
        now (datetime): datetime instance to be converted

    Returns:
        float: Excel style datetime
    """
    return int(f"{now.year:04d}{now.month:02d}{now.day:02d}") + round(
        (now.hour * 3600 + now.minute * 60 + now.second) / 86400, 6
    )


log = getLogger(__name__)
log.info(
    f"↓↓↓ JUSTUSE SESSION {excel_style_datetime(datetime.now())} ID:{sessionID} ↓↓↓"
)

# internal subpackage imports
test_version: str = locals().get("test_version")

_reloaders: dict["ProxyModule", "ModuleReloader"] = {}  # ProxyModule:Reloader


# sometimes all you need is a sledge hammer...
def _release_locks():
    for _ in range(2):
        [lock.unlock() for lock in threading._shutdown_locks]
        [reloader.stop() for reloader in _reloaders.values()]
    log.info(
        f"↑↑↑ JUSTUSE SESSION {excel_style_datetime(datetime.now())} ID:{sessionID} ↑↑↑"
    )


atexit.register(_release_locks)


class Use(ModuleType):
    """
    Welcome to the world of use

    """

    def __init__(self):
        # might run into issues during testing otherwise
        timer()
        self.registry = self._set_up_registry()
        "Registry sqlite DB to store all relevant package metadata."
        timer()
        if config.debugging:
            root.setLevel(DEBUG)

        if config.version_warning:
            try:
                timer()
                response = requests.get("https://pypi.org/pypi/justuse/json")
                "Checking if there's a new version of justuse."
                timer()
                data = response.json()
                max_version = max(
                    Version(version) for version in data["releases"].keys()
                )
                if Version(__version__) < max_version:
                    warn(
                        UserMessage.use_version_warning(max_version),
                        VersionWarning,
                    )
            except (KeyError, requests.exceptions.ConnectionError):
                if test_version:
                    raise
                log.error(
                    traceback.format_exc()
                )  # we really don't need to bug the user about this (either pypi is down or internet is broken)

    def clean_slate(self):
        shutil.rmtree(config.venv, ignore_errors=True)
        shutil.rmtree(config.packages, ignore_errors=True)
        config.venv.mkdir(mode=0o755, exist_ok=True)
        config.packages.mkdir(mode=0o755, parents=True, exist_ok=True)
        self.recreate_registry()

    def _set_up_registry(self, *, registry=None, path: Path | None = None):
        # recreating reuses the registry connection and file
        timer()
        if registry is None:
            if path or test_version and "DB_TEST" not in os.environ:
                registry = sqlite3.connect(path or ":memory:").cursor()
            else:
                try:
                    registry = sqlite3.connect(home / "registry.db").cursor()
                except Exception as e:
                    raise RuntimeError(UserMessage.couldnt_connect_to_db(e)) from e
        registry.row_factory = lambda cursor, row: {
            col[0]: row[idx] for idx, col in enumerate(cursor.description)
        }
        registry.execute("PRAGMA foreign_keys=ON")
        registry.execute("PRAGMA auto_vacuum = FULL")
        registry.executescript(
            """
CREATE TABLE IF NOT EXISTS "artifacts" (
	"id"    INTEGER,
	"distribution_id"   INTEGER,
	"import_relpath" TEXT,
	"artifact_path" TEXT,
    "module_path" TEXT,
	PRIMARY KEY("id" AUTOINCREMENT),
	FOREIGN KEY("distribution_id") REFERENCES "distributions"("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "distributions" (
	"id"    INTEGER,
	"name"  TEXT NOT NULL,
	"version"   TEXT NOT NULL,
	"installation_path" TEXT,
	"date_of_installation"  INTEGER,
	"number_of_uses"    INTEGER,
	"date_of_last_use"  INTEGER,
	"pure_python_package"   INTEGER NOT NULL DEFAULT 1,
	PRIMARY KEY("id" AUTOINCREMENT)
);

CREATE TABLE IF NOT EXISTS "hashes" (
	"algo"  TEXT NOT NULL,
	"value" INTEGER NOT NULL,
	"artifact_id"   INTEGER NOT NULL,
	PRIMARY KEY("algo","value"),
	FOREIGN KEY("artifact_id") REFERENCES "artifacts"("id") ON DELETE CASCADE
);
		"""
        )
        registry.connection.commit()
        timer()
        return registry

    def recreate_registry(self):
        number_of_backups = len(list(home.glob("registry.db*")))
        shutil.copyfile(
            home / "registry.db", home / f"registry.db.{number_of_backups}.bak"
        )
        self._clear_registry()
        self._set_up_registry(registry=self.registry)
        self.cleanup()

    def _clear_registry(self):
        for table in self.registry.execute(
            "SELECT name FROM sqlite_schema WHERE type='table';"
        ).fetchall():
            if table["name"] == "sqlite_sequence":
                continue
            self.registry.execute("DROP TABLE ?;", (table["name"],))
            self.registry.connection.commit()

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
            """
DELETE FROM hashes
WHERE artifact_id
IN (SELECT id FROM artifacts WHERE distribution_id IN (SELECT id FROM distributions WHERE name=? AND version=?))""",
            (name, str(version)),
        )
        self.registry.execute(
            "DELETE FROM artifacts WHERE distribution_id IN (SELECT id FROM distributions WHERE name=? AND version=?)",
            (name, str(version)),
        )
        self.registry.execute(
            "DELETE FROM distributions WHERE name=? AND version=?", (name, str(version))
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

        for name, version, artifact_path, installation_path in self.registry.execute("""
SELECT name, version, artifact_path, installation_path
FROM distributions
JOIN artifacts on distributions.id = distribution_id
""").fetchall():
            if not (
                _ensure_path(artifact_path).exists()
                and _ensure_path(installation_path).exists()
            ):
                self.del_entry(name, version)
        self.registry.connection.commit()

    @singledispatchmethod
    def __call__(self, thing, /, *args, **kwargs):
        raise NotImplementedError(UserMessage.cant_use(thing))

    @require(lambda hash_algo: hash_algo in Hash)
    @require(lambda as_import: as_import.isidentifier())
    @__call__.register
    def _use_url(
        self,
        url: URL,
        /,
        *,
        hash_algo=Hash.sha256,
        hash_value=None,
        initial_globals: dict[Any, Any] | None = None,
        import_as: str = None,
        default=Modes.fastfail,
        modes=0,
    ) -> ProxyModule:
        """
        Import a module from a web source.

        >>> load = use(
                use.URL("https://raw.githubusercontent.com/amogorkon/stay/master/src/stay/stay.py"), modes=use.recklessness
                , import_as="stay").Decoder()
        >>> for x in load("a: b"): x
        {'a': 'b'}

        Args:
            url (URL): a web url, wrapped with use.URL()
            hash_algo (_type_, optional): Hash algo used to check. Defaults to Hash.sha256.
            hash_value (_type_, optional): Hash value used to pin the content. Defaults to None.
            initial_globals (Optional[dict[Any, Any]], optional): Any globals passed into the module. Defaults to None.
            import_as (str, optional): Valid identifier which should be used for "importing" -
                means the module can be imported anywhere else using this name. Defaults to None.
            default (_type_, optional): Any value (like a different module) in case importing fails. Defaults to Modes.fastfail.
            modes (int, optional):
                * use.recklessness - to skip hash validation


        Raises:
            ImportError: If no default is given, return ImportError if the module cannot be imported

        Returns:
            ProxyModule: the module wrapped with use.ProxyModule for convenience
        """
        if import_as:
            if (mod := _import_as(import_as)) is not None:
                return mod

        log.debug(f"use-url: {url}")
        name = url.path.segments[-1]

        if query := self.registry.execute(
            "SELECT module_path FROM artifacts WHERE artifact_path=?",
            (str(url),),
        ).fetchone():
            module_path = Path(query["module_path"])
        else:
            module_path = None

        if module_path is None or not Path(module_path).exists():
            self.registry.execute(
                "DELETE FROM artifacts WHERE artifact_path=?", (str(url),)
            )
            self.registry.connection.commit()
            for p in config.web_modules.glob(f"*_{name}"):
                p.unlink()
            # pyc and other shenanigans
            for p in config.web_modules.glob(f"*_{name}?"):
                p.unlink()
        elif (module_path.parent / f"{module_path.name}c").exists():
            module_path = module_path.parent / f"{module_path.name}c"
            try:
                mod = ProxyModule(module_from_pyc(name, module_path, initial_globals))
                return ProxyModule(mod)
            except:
                raise
        else:
            content = Path(module_path).read_bytes()
            # compile the module into a pyc file and save it for next time
            py_compile.compile(
                module_path, module_path.parent / f"{module_path.name}c", optimize=2
            )

        if not content:
            response = requests.get(str(url))
            if response.status_code != 200:
                raise ImportError(UserMessage.web_error(url, response))
            content = response.content

            this_hash = hash_algo.value(content).hexdigest()

            if not Modes.recklessness & modes:
                if hash_value:
                    if this_hash != hash_value:
                        return _fail_or_default(
                            UnexpectedHash(
                                f"{this_hash} does not match the expected hash {hash_value} - aborting!"
                            ),
                            default,
                        )
                else:
                    warn(
                        UserMessage.no_validation(url, hash_algo, this_hash),
                        NoValidationWarning,
                    )

            module_path = (
                config.web_modules / f"{excel_style_datetime(datetime.now())}_{name}"
            )
            module_path.touch(mode=0o755)
            module_path.write_bytes(content)
            py_compile.compile(
                module_path, module_path.parent / f"{module_path.name}c", optimize=2
            )

            self.registry.execute(
                """
INSERT OR IGNORE INTO artifacts (artifact_path, module_path)
VALUES (?, ?)
""",
                (str(url), str(module_path)),
            )
            self.registry.connection.commit()

        try:
            mod = _build_mod(
                mod_name=import_as or name,
                code=content,
                module_path=module_path,
                initial_globals=initial_globals,
            )
        except KeyError:
            raise
        if exc := None:
            return _fail_or_default(ImportError(exc), default)
        mod = ProxyModule(mod)
        return mod

    @__call__.register
    def _use_git(
        self,
        git_repo: git,
        /,
        *,
        modes=0,
    ) -> ProxyModule:
        """Install git repo."""

    @__call__.register
    def _use_path(
        self,
        path: Path,
        /,
        *,
        initial_globals=None,
        import_as: str = None,
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
        initial_globals = initial_globals or {}
        if import_as:
            assert import_as not in sys.modules

        reloading = bool(Use.reloading & modes)

        exc = None
        mod = None
        original_cwd = Path.cwd()

        if path.is_dir():
            return _fail_or_default(
                ImportError(f"Can't import directory {path}"), default
            )

        try:
            name, mod_name, pkg_name, path = _real_path(
                path=path,
                _applied_decorators=_applied_decorators,
                landmark=Use.__call__.__code__,
            )
        except (NotImplementedError, ImportError):
            exc = traceback.format_exc()
        sys.path.append(path.parent)

        with open(path, "rb") as rfile:
            code = rfile.read()
        try:
            mod = _build_mod(
                mod_name=mod_name,
                code=code,
                initial_globals=initial_globals,
                module_path=path,
                pkg_name=pkg_name,
            )
        except KeyError:
            exc = traceback.format_exc()
        if exc:
            return _fail_or_default(exc, default)
        mod = ProxyModule(mod)

        if reloading:
            reloader = ModuleReloader(
                proxy=mod,
                name=name,
                path=path,
                initial_globals=initial_globals,
                pkg_name=pkg_name,
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
                warn(UserMessage.not_reloadable(name), NotReloadableWarning)

        os.chdir(original_cwd)
        if exc:
            return _fail_or_default(ImportError(exc), default)

        if import_as:
            sys.modules[import_as] = mod
            if len(import_as.split(".")) > 1:
                mod.__package__ = import_as.split()[0]
        return mod

    @__call__.register
    def _use_kwargs(
        self,
        thing: None,  # sic! otherwise single-dispatch with 'empty' *args won't work
        /,
        *,
        pkg_name: str = None,
        mod_name: str = None,
        version: Version | str | None = None,
        hash_algo=Hash.sha256,
        hashes: str | list[str] | None = None,
        default=Modes.fastfail,
        modes: int = 0,
        import_as: str = None,
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
            ProxyModule|Any: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-kwargs: {pkg_name} {mod_name} {version} {hashes}")
        return self._use_package(
            name=f"{pkg_name}/{mod_name}",
            pkg_name=pkg_name,
            mod_name=mod_name,
            version=Version(version) if version else None,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
            Message=KwargMessage,
            import_as=import_as,
        )

    @__call__.register
    def _use_tuple(
        self,
        pkg_tuple: tuple,
        /,
        *,
        version: Version | str | None = None,
        hash_algo=Hash.sha256,
        hashes: str | list[str] | None = None,
        default=Modes.fastfail,
        modes: int = 0,
        import_as: str = None,
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
            ProxyModule|Any: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-tuple: {pkg_tuple} {version} {hashes}")
        pkg_name, mod_name = pkg_tuple
        return self._use_package(
            name=f"{pkg_name}/{mod_name}",
            pkg_name=pkg_name,
            mod_name=mod_name,
            version=Version(version) if version else None,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
            Message=TupleMessage,
            import_as=import_as,
        )

    @__call__.register
    def _use_str(
        self,
        name: str,
        /,
        *,
        version: Version | str | None = None,
        hash_algo=Hash.sha256,
        hashes: str | list[str] | None = None,
        default=Modes.fastfail,
        modes: int = 0,
        import_as: str = None,
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
        pkg_name, mod_name = _parse_name(name)
        return self._use_package(
            name=name,
            pkg_name=pkg_name,
            mod_name=mod_name,
            req_ver=Version(version) if version else None,
            hash_algo=hash_algo,
            hashes=hashes,
            default=default,
            modes=modes,
            Message=StrMessage,
            import_as=import_as,
        )

    @require(lambda hash_algo: hash_algo is not None)
    def _use_package(
        self,
        *,
        name,
        pkg_name: str,
        mod_name: str,
        req_ver: Version | None,
        hashes: str | set[str] | None,
        default: Any,
        hash_algo: Hash,
        modes: int = 0,
        Message: type = UserMessage,
        import_as: str = None,
    ):
        # preparing the yummy kwargs for the buffet...
        auto_install = bool(Modes.auto_install & modes)
        no_public_installation = bool(Modes.no_public_installation & modes)
        fastfail = bool(Modes.fastfail & modes)
        fatal_exceptions = bool(Modes.fatal_exceptions & modes)
        no_browser = bool(Modes.no_browser & modes)
        cleanup = not bool(Modes.no_cleanup & modes)
        hashes: set[int] = _hashes(hashes)

        if mod_name:
            mod_name = mod_name.replace("/", ".").replace("-", "_")

        # let's see what we'll get from the buffet table
        case = (
            bool(req_ver),
            bool(hashes),
            (installed_version := _installed_version(pkg_name)) is not None
            or _is_builtin(pkg_name),
            auto_install,
        )

        log.info(
            f"{name=}, {pkg_name=}, {mod_name=}, {hashes=}, {req_ver=}, {installed_version=}, {auto_install=}, {case=}"
        )
        # welcome to the buffet table, where everything is a lie
        kwargs = {
            "name": name,
            "pkg_name": pkg_name,
            "mod_name": mod_name,
            "req_ver": req_ver,
            "user_provided_hashes": hashes,
            "hash_algo": hash_algo,
            "fastfail": fastfail,
            "no_public_installation": no_public_installation,
            "fatal_exceptions": fatal_exceptions,
            "sys_version": Version(".".join(map(str, sys.version_info[:3]))),
            "no_browser": no_browser,
            "Message": Message,
            "registry": self.registry,
            "cleanup": cleanup,
            "installed_version": installed_version,
        }

        result = buffet_table(case, kwargs)
        assert result
        assert isinstance(result, (Exception, ModuleType))

        if isinstance(result, Exception):
            return _fail_or_default(result, default)

        if isinstance(result, ModuleType):
            if import_as:
                M = sys.modules[mod_name]
                sys.modules[import_as] = M
                del sys.modules[mod_name]

            return ProxyModule(result)


def excel_style_datetime(now: datetime) -> float:
    """
    Build a float representing the current time in the excel format.
    First 4 digits are the year, the next two are the month, the next two are the day followed
    by a decimal point, then time in fraction of the day.
    Args:
        now (datetime): datetime instance to be converted
    Returns:
        float: Excel style datetime
    """
    return int(f"{now.year:04d}{now.month:02d}{now.day:02d}") + round(
        (now.hour * 3600 + now.minute * 60 + now.second) / 86400, 6
    )


def _hashes(hashes: str | list[str] | None) -> set[int]:
    # a single hash is a string
    if isinstance(hashes, str):
        # spaces are hard to see in JACK, so we ignore them
        res = ["".join(hashes.split())]
    if not hashes:
        res = set()
    else:
        res: set[int] = {
            JACK_as_num(H) if is_JACK(H) else int(H, 16)
            for H in ("".join(H.split()) for H in hashes)
        }
    return res


def _installed_version(pkg_name: str) -> Version | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return Version(importlib.metadata.version(pkg_name))
    return None
