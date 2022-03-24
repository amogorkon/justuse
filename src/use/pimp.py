"""
Delegating package installation to pip, packaging and friends.
"""



import codecs
import contextlib
import importlib.util
import linecache
import os
import platform
import re
import sys
import tarfile
import time
import traceback
import zipfile
import zipimport
from collections.abc import Callable
from functools import lru_cache as cache
from functools import reduce
from importlib import metadata
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.metadata import Distribution, PackageNotFoundError, PackagePath
from itertools import chain, product
from logging import getLogger
from os.path import abspath, split
from pathlib import Path, PureWindowsPath, WindowsPath
from shutil import rmtree
from sqlite3 import Cursor
from subprocess import CalledProcessError, run
from types import ModuleType
from typing import (Any, Iterable, Optional, Protocol, TypeVar, Union,
                    runtime_checkable)
from warnings import catch_warnings, filterwarnings, warn

import furl
import packaging
import requests
from beartype import beartype
from furl import furl as URL
from icontract import ensure, require
from packaging import tags
from packaging.specifiers import SpecifierSet

from use import (Hash, InstallationError, Modes, UnexpectedHash,
                 VersionWarning, config, sessionID)
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest
from use.messages import (UserMessage, _web_pebkac_no_hash,
                          _web_pebkac_no_version_no_hash)
from use.pydantics import PyPI_Project, PyPI_Release, RegistryEntry, Version
from use.tools import pipes

log = getLogger(__name__)


T = TypeVar("T")


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


@beartype
def _ensure_version(
    result: Union[ModuleType, Exception], *, name, version, **kwargs
) -> Union[ModuleType, Exception]:
    if not isinstance(result, ModuleType):
        return result
    result_version = _get_version(mod=result)
    if result_version != version:
        warn(UserMessage.version_warning(name, version, result_version), category=VersionWarning)
    return result


# fmt: off
@pipes
def _ensure_path(value: Union[bytes, str, furl.Path, Path]) -> Path:
    if isinstance(value, (str, bytes)):
        return Path(value).absolute()
    if isinstance(value, furl.Path):
        return (
            Path.cwd(),
            value.segments
            << map(Path)
            << tuple
            << reduce(Path.__truediv__),
        ) << reduce(Path.__truediv__)
    return value
# fmt: on


@cache
@beartype
def get_supported() -> frozenset[PlatformTag]:  # cov: exclude
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, package_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    get_supported = None
    with catch_warnings():
        filterwarnings(action="ignore", category=DeprecationWarning)
        with contextlib.suppress(ImportError):
            from pip._internal.resolution.legacy.resolver import get_supported
        if not get_supported:
            with contextlib.suppress(ImportError):
                from pip._internal.models.target_python import get_supported
        if not get_supported:
            with contextlib.suppress(ImportError):
                from pip._internal.utils.compatibility_tags import \
                    get_supported
        if not get_supported:
            with contextlib.suppress(ImportError):
                from pip._internal.resolution.resolvelib.factory import \
                    get_supported
    get_supported = get_supported or (lambda: [])

    items: list[PlatformTag] = [PlatformTag(platform=tag.platform) for tag in get_supported()]

    items.extend(PlatformTag(platform=str(tag)) for tag in packaging.tags._platform_tags())

    return frozenset(items)


@beartype
def _filter_by_version(releases: list[PyPI_Release], *, version: Version) -> list[PyPI_Release]:
    return list(filter(lambda r: r.version == version, releases))


class ZipFunctions:
    def __init__(self, artifact_path):
        self.archive = zipfile.ZipFile(artifact_path)

    def get(self):
        return (self.archive, [e.filename for e in self.archive.filelist])

    def read_entry(self, entry_name):
        with self.archive.open(entry_name) as f:
            bdata = f.read()
            text = bdata.decode("ISO-8859-1").splitlines() if len(bdata) < 8192 else ""
            return (Path(entry_name).stem, text)


class TarFunctions:
    def __init__(self, artifact_path):
        self.archive = tarfile.open(artifact_path)

    def get(self):
        return (self.archive, [m.name for m in self.archive.getmembers() if m.type == b"0"])

    def read_entry(self, entry_name):
        m = self.archive.getmember(entry_name)
        with self.archive.extractfile(m) as f:
            bdata = f.read()
            text = bdata.decode("UTF-8").splitlines() if len(bdata) < 8192 else ""
            return (Path(entry_name).stem, text)


@beartype
@pipes
def archive_meta(artifact_path):
    DIST_PKG_INFO_REGEX = re.compile("(dist-info|-INFO|\\.txt$|(^|/)[A-Z0-9_-]+)$")

    if ".tar" in Path(str(artifact_path)).stem:
        functions = TarFunctions(artifact_path)
    else:
        functions = ZipFunctions(artifact_path)

    archive, names = functions.get()
    meta = names << filter(DIST_PKG_INFO_REGEX.search) << map(functions.read_entry) >> dict
    meta.update(
        dict(
            (lp := l.partition(": "), (lp[0].lower().replace("-", "_"), lp[2]))[-1]
            for l in meta.get("METADATA", meta.get("PKG-INFO"))
            if ": " in l
        )
    )
    name = meta.get("name", Path(artifact_path).stem.split("-")[0])
    meta["name"] = name
    if "top_level" not in meta:
        meta["top_level"] = [""]
    (
        top_level,
        name,
    ) = (meta["top_level"][0], meta["name"])
    import_name = name if top_level == name else ".".join((top_level, name))
    meta["names"] = names
    meta["import_name"] = import_name
    for relpath in sorted(
        [n for n in names if len(n) > 4 and n[-3:] == ".py"],
        key=lambda n: (
            not n.startswith(import_name),
            not n.endswith("__init__.py"),
            len(n),
        ),
    ):
        meta["import_relpath"] = relpath
        break
    else:
        meta["import_relpath"] = f"{import_name}.py"
    archive.close()
    return meta


@beartype
def _clean_sys_modules(package_name: str) -> None:
    for k in dict(
        [
            (k, v)
            for k, v in list(sys.modules.items())
            if package_name in k.split(".")
            and (
                getattr(v, "__spec__", None) is None
                or isinstance(v, (SourceFileLoader, zipimport.zipimporter))
            )
        ]
    ):
        if k in sys.modules:
            del sys.modules[k]


@beartype
def _pebkac_no_version(
    *,
    name: str,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    Message: type,
    **kwargs,
) -> Union[ModuleType, Exception]:

    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
        assert False, f"{func}() returned {result!r}"

    return RuntimeWarning(Message.cant_import_no_version(name))


@beartype
@pipes
def _pebkac_no_hash(
    *,
    name: str,
    version: Version,
    package_name: str,
    no_browser: bool,
    Message: type,
    hash_algo: Hash,
    **kwargs,
) -> RuntimeWarning:
    project = _get_data_from_pypi(package_name=package_name, version=version)
    releases = _get_releases(project)

    if version not in (r.version for r in releases):
        last_version = next(iter(reversed(project.releases)))
        return RuntimeWarning(Message.no_distribution_found(package_name, version, last_version))

    project = _filter_by_version(releases, version=version)
    filtered = _filter_by_platform(
        releases,
        tags=get_supported(),
    )
    ordered = _sort_releases(filtered)

    if not no_browser:
        _web_pebkac_no_hash(name=name, package_name=package_name, version=version, releases=ordered)

    if not ordered:
        return RuntimeWarning(Message.no_recommendation(package_name, version))

    recommended_hash = hexdigest_as_JACK(ordered[0].digests.get(hash_algo.name))

    # for test_suggestion_works, don't remove
    print(recommended_hash)

    return RuntimeWarning(
        Message.pebkac_missing_hash(
            name=name,
            package_name=package_name,
            version=version,
            recommended_hash=recommended_hash,
            no_browser=no_browser,
        )
    )


@beartype
@pipes
def _pebkac_no_version_no_hash(
    *,
    name: str,
    package_name: str,
    no_browser: bool,
    Message: type,
    **kwargs,
) -> Exception:
    # let's try to make an educated guess and give a useful suggestion
    proj = _get_data_from_pypi(package_name=package_name)
    releases = _get_releases(proj)

    ordered = releases >> _filter_by_platform(tags=get_supported()) >> _sort_releases
    # we tried our best, but we didn't find anything that could work

    # let's try to find *anything*
    if not ordered:
        ordered = _sort_releases(releases)
        if not ordered:
            # we tried our best..
            return RuntimeWarning(Message.pebkac_unsupported(package_name))

        if not no_browser:
            version = ordered[-1].version
            _web_pebkac_no_hash(name=name, package_name=package_name, version=version, project=proj)
            return RuntimeWarning(
                Message.pebkac_no_version_no_hash(
                    name=name, package_name=package_name, version=version, no_browser=no_browser
                )
            )

    recommended_version = ordered[0].version
    # for test_suggestion_works, don't remove (unless you want to work out the details)
    print(recommended_version)

    # we found something that could work, but it may not fit to the user's requirements
    return RuntimeWarning(
        Message.pebkac_no_version_no_hash(
            name=name,
            package_name=package_name,
            version=recommended_version,
            no_browser=no_browser,
        )
    )


@beartype
def _import_public_no_install(
    *,
    module_name: str,
    **kwargs,
) -> Union[Exception, ModuleType]:
    # builtin?
    builtin = False
    try:
        metadata.PathDistribution.from_name(module_name)
    except metadata.PackageNotFoundError:  # indeed builtin!
        builtin = True
    # TODO ehhh.. builtin needs to be handled differently?

    mod = sys.modules.get(module_name)
    imported = bool(mod)

    if not mod:
        mod = importlib.import_module(module_name)

    # # ? does this solve the caching issue?
    # if not imported:
    #     del sys.modules[module_name]
    return mod


def _parse_name(name: str) -> tuple[str, str]:
    """Parse the user-provided name into a package name for installation and a module name for import.

    The package name is whatever pip would expect.
    The module name is whatever import would expect.

    Mini-DSL: / separates the package name from the module name.
    """
    if not name:
        return None, None

    def old():
        # as fallback
        match = re.match(r"(?P<package_name>[^/.]+)/?(?P<rest>[a-zA-Z0-9._]+)?$", name)
        assert match, f"Invalid name spec: {name!r}"
        names = match.groupdict()
        package_name = names["package_name"]
        rest = names["rest"]
        if not package_name:
            package_name = rest
        if not rest:
            rest = package_name
        return package_name

    package_name = None
    module_name = ""
    if "/" in name:
        if name.count("/") > 1:
            raise ImportError(
                f"Invalid name spec: {name!r}, can't have multiple / characters as package/module separator."
            )
        package_name, _, module_name = name.partition("/")
    else:
        package_name = old()
        module_name = name
    return (package_name, module_name)


@beartype
def _check_db_for_installation(*, registry=Cursor, package_name=str, version) -> Optional[RegistryEntry]:
    query = registry.execute(
        """
        SELECT
            artifact_path, installation_path, pure_python_package
        FROM distributions
        JOIN artifacts ON artifacts.id = distributions.id
        WHERE name=? AND version=?
        ORDER BY artifacts.id DESC
        """,
        (
            package_name,
            str(version),
        ),
    ).fetchone()
    return RegistryEntry(**query) if query else None


@beartype
def _auto_install(
    *,
    package_name: str,
    module_name: str,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    version: Version,
    hash_algo: Hash,
    user_provided_hashes: set[int],
    registry: Cursor,
    cleanup: bool,
    **kwargs,
) -> Union[ModuleType, BaseException]:
    """Install, if necessary, the package and import the module in any possible way."""
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
        else:
            raise AssertionError(f"{func!r} returned {result!r}")

    if entry := _check_db_for_installation(registry=registry, package_name=package_name, version=version):
        # is there a point in checking the hashes at this point? probably not.
        if entry.pure_python_package:
            assert entry.artifact_path.exists()
            # let's not try to catch this - since this apparently already worked *at least once* - no fallback
            return zipimport.zipimporter(entry.artifact_path).load_module(module_name)
        # else: we have an installed package, let's try to import it
        original_cwd = Path.cwd()
        assert entry.installation_path.exists()
        os.chdir(entry.installation_path)
        # with an installed package there may be weird issues we can't be sure about so let's be safe
        try:
            return _load_venv_entry(
                module_name=module_name,
                installation_path=entry.installation_path,
            )
        except BaseException as err:
            traceback.print_exc(file=sys.stderr)
            msg = err
        finally:
            os.chdir(original_cwd)
        return ImportError(msg)

    # else: we have to download the package and install it
    project = _get_data_from_pypi(package_name=package_name, version=version)
    # we *did* ask the user to give us hashes of artifacts that *should* work, so let's check for those.
    # We can't be sure which one of those hashes will work on this platform, so let's try all of them.

    for H in user_provided_hashes:
        log.info(f"Attempting auto-installation of <{num_as_hexdigest(H)}>...")
        url = next(
            (URL(purl.url) for purl in project.urls if H == int(purl.digests[hash_algo.name], 16)),
            None,
        )

        if not url:
            return UnexpectedHash(
                f"user provided {user_provided_hashes}, do not match any of the {len(project.urls)} possible artifacts"
            )

        # got an url for an artifact with a hash given by the user, let's install it
        filename = url.asdict()["path"]["segments"][-1]
        artifact_path = config.packages / filename
        _download_artifact(artifact_path=artifact_path, url=url, hash_value=H, hash_algo=hash_algo)
        try:
            log.info("Attempting to install..")
            entry = _install(
                package_name=package_name,
                artifact_path=artifact_path,
                version=version,
                force_install=True,
            )
            # packages like tensorflow-gpu only need to be installed but nothing imported, so module name is empty
            log.info("Attempting to import...")
            mod = _load_venv_entry(
                module_name=module_name,
                installation_path=entry.installation_path,
            )
            log.info("Successfully imported.")
        except BaseException as err:
            msg = err  # sic
            log.error(err)
            traceback.print_exc(file=sys.stderr)
            if entry and cleanup:
                rmtree(entry.installation_path)
                assert not entry.installation_path.exists()
            continue

        _save_package_info(
            package_name=package_name,
            version=version,
            artifact_path=entry.artifact_path,
            hash_value=int(hash_algo.value(entry.artifact_path.read_bytes()).hexdigest(), 16),
            hash_algo=hash_algo,
            installation_path=entry.installation_path,
            registry=registry,
        )
        return mod
    log.critical(
        f"Could not install {package_name!r} {version!r}. Hashes that were attempted: {[num_as_hexdigest(H) for H in user_provided_hashes]}"
    )
    return ImportError(msg)


@beartype
def _save_package_info(
    *,
    registry=Cursor,
    version: Version,
    artifact_path: Path,
    installation_path: Path,
    hash_value=int,
    hash_algo: Hash,
    package_name: str,
):
    """Update the registry to contain the pkg's metadata."""
    if not registry.execute(
        f"SELECT * FROM distributions WHERE name='{package_name}' AND version='{version}'"
    ).fetchone():
        registry.execute(
            f"""
INSERT INTO distributions (name, version, installation_path, date_of_installation, pure_python_package)
VALUES ('{package_name}', '{version}', '{installation_path}', {time.time()}, {installation_path is None})
"""
        )
        registry.execute(
            f"""
INSERT OR IGNORE INTO artifacts (distribution_id, artifact_path)
VALUES ({registry.lastrowid}, '{artifact_path}')
"""
        )
        registry.execute(
            f"""
INSERT OR IGNORE INTO hashes (artifact_id, algo, value)
VALUES ({registry.lastrowid}, '{hash_algo.name}', '{hash_value}')"""
        )
    registry.connection.commit()


@beartype
@ensure(lambda url: str(url).startswith("http"))
def _download_artifact(*, artifact_path: Path, url: URL, hash_algo: Hash, hash_value: int):
    # let's check if we downloaded it already, just in case
    if (
        artifact_path.exists()
        and int(hash_algo.value(artifact_path.read_bytes()).hexdigest(), 16) == hash_value
    ):
        log.info("Artifact already downloaded. Hashes matching.")
        return
    # this should work since we just got the url from pypi itself - if this fails, we have bigger problems
    log.info("Downloading artifact from PyPI...")
    data = requests.get(url).content
    artifact_path.write_bytes(data)
    if int(hash_algo.value(artifact_path.read_bytes()).hexdigest(), 16) != hash_value:
        # let's try once again, cosmic rays and all, believing in pure dumb luck
        log.info("Artifact downloaded but hashes don't match. Trying again...")
        data = requests.get(url).content
        artifact_path.write_bytes(data)
    if int(hash_algo.value(artifact_path.read_bytes()).hexdigest(), 16) != hash_value:
        # this means either PyPI is hacked or there is a man-in-the-middle
        raise ImportError("Hashes don't match. Aborting. Something very fishy is going on.")
    log.info("Download successful.")
    return


@beartype
def _is_pure_python_package(artifact_path: Path, meta: dict) -> bool:
    for n, s in product(meta["names"], importlib.machinery.EXTENSION_SUFFIXES):
        if n.endswith(s):
            return False
    if ".tar" in str(artifact_path):
        return False
    return True


@beartype
def _find_module_in_venv(package_name: str, version: Version, relp: str) -> Path:
    env_dir = config.venv / package_name / str(version)
    log.debug("env_dir=%s", env_dir)
    site_dirs = [
        env_dir / f"Lib{suffix}" / "site-packages"
        if sys.platform == "win32"
        else env_dir / f"lib{suffix}" / ("python%d.%d" % sys.version_info[:2]) / "site-packages"
        for suffix in ("64", "")
    ]
    log.debug("site_dirs=%s", site_dirs)
    for p in env_dir.glob("**/*"):
        log.debug("  - %s", p.relative_to(env_dir).as_posix())

    original_sys_path = sys.path
    try:
        # Need strings for sys.path to work
        sys.path = [*map(str, site_dirs), *sys.path]
        sys.path_importer_cache.clear()
        importlib.invalidate_caches()
        # sic! importlib uses sys.path for lookup
        dist = Distribution.from_name(package_name)
        log.info("dist=%s", dist)
        log.debug("dist.files=%s", dist.files)
        path_set = dist.files
        log.debug("path_set=%s", path_set)
        for path in path_set:
            log.debug("path=%s", path)
            file = dist.locate_file(path)
            log.debug("file=%s", file)
            if not file.exists():
                continue
            real_file = Path(*file.parts[: -len(path.parts)])
            log.debug("real_file=%s", real_file)
            if real_file.exists():
                return real_file

    finally:
        sys.path = original_sys_path
    raise ImportError("No module in site_dirs")


@beartype
def _install(
    *,
    package_name: str,
    version: Version = None,
    force_install=False,
    artifact_path: Path,
) -> RegistryEntry:
    """Take care of the installation."""
    meta = archive_meta(artifact_path)
    import_parts = re.split("[\\\\/]", meta["import_relpath"])
    if "__init__.py" in import_parts:
        import_parts.remove("__init__.py")
    relp: str = meta["import_relpath"]
    venv_root = config.venv / package_name / str(version)
    site_pkgs_dir = list(venv_root.rglob("site-packages"))
    if not any(site_pkgs_dir):
        force_install = True
        module_paths = []
    else:
        installation_path = site_pkgs_dir[0]
        module_paths = venv_root.rglob(f"**/{relp}")

    python_exe = Path(sys.executable)

    if not module_paths or force_install:
        # If we get here, the venv/pip setup is required.
        # we catch errors one level higher, so we don't have to deal with them here
        env = {}
        _realenv = {
            k: v
            for k, v in chain(os.environ.items(), env.items())
            if isinstance(k, str) and isinstance(v, str)
        }

        argv = [
            python_exe,
            "-m",
            "pip",
            "--disable-pip-version-check",
            "--no-color",
            "--verbose",
            "--verbose",
            "install",
            "--pre",
            "--root",
            PureWindowsPath(venv_root).drive
            if isinstance(venv_root, (WindowsPath, PureWindowsPath))
            else "/",
            "--prefix",
            str(venv_root),
            "--progress-bar",
            "ascii",
            "--prefer-binary",
            "--exists-action",
            "i",
            "--ignore-installed",
            "--no-warn-script-location",
            "--force-reinstall",
            "--no-warn-conflicts",
            artifact_path,
        ]
        try:
            setup = dict(
                executable=python_exe,
                args=[*map(str, argv)],
                bufsize=1024,
                input="",
                capture_output=False,
                timeout=45000,
                check=True,
                close_fds=True,
                env=_realenv,
                encoding="ISO-8859-1",
                errors="ISO-8859-1",
                text=True,
                shell=False,
            )
        except CalledProcessError as err:
            log.error("::".join(err.cmd, err.output, err.stdout, err.stderr))
            raise InstallationError(err) from err
        output = run(**setup)
        log.info("Installation successful.")

    installation_path = _find_module_in_venv(package_name=package_name, version=version, relp=relp)

    return RegistryEntry(
        installation_path=installation_path,
        artifact_path=artifact_path,
        pure_python_package=False,
    )


@beartype
def _load_venv_entry(*, module_name: str, installation_path: Path) -> ModuleType:
    if not module_name:
        log.info("Module name is empty, returning empty Module.")
        return ModuleType("")
    origcwd = Path.cwd()
    original_sys_path = list(sys.path)
    # TODO we need to keep track of package-specific sys-paths
    if sys.path[0] != "":
        sys.path.insert(0, "")
    try:
        os.chdir(installation_path)
        # importlib and sys.path.. bleh
        return importlib.import_module(module_name)
    except BaseException as err:
        msg = err
        log.error(msg)
        traceback.print_exc(file=sys.stderr)
    finally:
        os.chdir(origcwd)
        sys.path = original_sys_path
    raise ImportError(msg)


@cache(maxsize=512)
def _get_data_from_pypi(*, package_name: str, version: str = None) -> PyPI_Project:
    if version:
        json_url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    else:
        json_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(json_url)
    if response.status_code == 404:
        raise ImportError(UserMessage.pebkac_unsupported(package_name))
    elif response.status_code != 200:
        raise RuntimeWarning(UserMessage.web_error(json_url, response))
    return PyPI_Project(**response.json())


@beartype
def _filter_by_platform(releases: list[PyPI_Release], *, tags: frozenset[PlatformTag]) -> list[PyPI_Release]:
    filtered = [
        release for release in releases if _is_compatible(release, platform_tags=tags, include_sdist=False)
    ]
    if not filtered:
        return [
            release for release in releases if _is_compatible(release, platform_tags=tags, include_sdist=True)
        ]
    return filtered


@beartype
def _get_releases(project: PyPI_Project) -> list[PyPI_Release]:
    return reduce(list.__add__, project.releases.values(), [])


@beartype
@pipes
def _sort_releases(releases: list[PyPI_Release]) -> list[PyPI_Release]:
    return sorted(
        releases,
        key=(
            lambda r: (
                1 - int(r.filename.endswith(".tar.gz")),
                1 - int(r.is_sdist),
                r.version,
            )
        ),
        reverse=True,
    )


@cache
def _is_version_satisfied(specifier: str, sys_version) -> bool:
    """
    SpecifierSet("") matches anything, no need to artificially
    lock down versions at this point

    @see https://warehouse.readthedocs.io/api-reference/json.html
    @see https://packaging.pypa.io/en/latest/specifiers.html
    """
    specifiers = SpecifierSet(specifier or "")
    return not specifier or sys_version in specifiers


@beartype
@pipes
def _is_platform_compatible(
    info: PyPI_Release, platform_tags: frozenset[PlatformTag], include_sdist=False
) -> bool:

    if not include_sdist and (".tar" in info.justuse.ext or info.justuse.python_tag in ("cpsource", "sdist")):
        return False

    if "win" in (info.packagetype or "unknown") and sys.platform != "win32":
        return False

    if info.platform_tag:
        if "win32" in info.platform_tag and sys.platform != "win32":
            return False
        if "macosx" in info.platform_tag and sys.platform != "darwin":
            return False

    our_python_tag = tags.interpreter_name() + tags.interpreter_version()
    supported_tags = {
        our_python_tag,
        "py3",
        "cp3",
        f"cp{tags.interpreter_version()}",
        f"py{tags.interpreter_version()}",
    }


    if info.platform_tag:
        given_platform_tags = info.platform_tag.split(".") << map(PlatformTag) >> frozenset
    else:
        return include_sdist

    if info.is_sdist and info.requires_python:
        given_python_tag = {
            our_python_tag
            for p in info.requires_python.split(",")
            if Version(platform.python_version()) in SpecifierSet(p)
        }
    else:
        given_python_tag = set(info.python_tag.split("."))

    return any(supported_tags.intersection(given_python_tag)) and (
        (info.is_sdist and include_sdist) or any(given_platform_tags.intersection(platform_tags))
    )


@beartype
def _is_compatible(info: PyPI_Release, platform_tags, include_sdist=False) -> bool:
    return (
        _is_platform_compatible(info, platform_tags, include_sdist)
        and not info.yanked
        and (include_sdist or info.justuse.ext not in ("tar", "tar.gz" "zip"))
    )


@beartype
def _get_version(name: Optional[str] = None, package_name=None, /, mod=None) -> Optional[Version]:
    version: Optional[Union[Callable[...], Version, Version, str]] = None
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
        version = version()
    if isinstance(version, str):
        return Version(version)
    return Version(version)


def _build_mod(
    *,
    module_name,
    code,
    initial_globals: Optional[dict[str, Any]],
    module_path,
    package_name="",
) -> ModuleType:
    mod = ModuleType(module_name)
    log.info(f"{Path.cwd()=} {package_name=} {module_name=} {module_path=}")
    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__path__ = [str(module_path.parent)]
    mod.__package__ = package_name
    mod.__name__ = module_name
    loader = SourceFileLoader(module_name, str(module_path))
    mod.__loader__ = loader
    mod.__spec__ = ModuleSpec(module_name, loader)
    sys.modules[module_name] = mod
    code_text = codecs.decode(code)
    # module file "<", ">" chars are specially handled by inspect
    getattr(linecache, "cache")[module_path] = (
        len(code),  # size of source code
        None,  # last modified time; None means there is no physical file
        [
            *map(lambda ln: ln + "\x0a", code_text.splitlines())
        ],  # a list of lines, including trailing newline on each
        mod.__file__,  # file name, e.g. "<mymodule>" or the actual path to the file
    )
    # not catching this causes the most irritating bugs ever!
    try:
        codeobj = compile(code, module_path, "exec")
        exec(codeobj, mod.__dict__)
    except:  # reraise anything without handling - clean and simple.
        raise
    return mod


def _fail_or_default(exception: BaseException, default: Any):
    if default is not Modes.fastfail:
        return default
    else:
        raise exception


# ### active web dev ###
# proj = None
# from shutil import copy


# def _qwer():
#     copy(Path(__file__).absolute().parent / r"templates/stylesheet.css", config.home / "stylesheet.css")
#     package_name = "pygame"
#     name = "pygame/foo"
#     version = Version("2.1.0")
#     global proj
#     if proj is None:
#         proj = _get_data_from_pypi(package_name=package_name, version=version)
#         proj = _filter_by_version(proj, version)

#     _web_pebkac_no_hash(package_name=package_name, version=version, name=name, project=proj)
