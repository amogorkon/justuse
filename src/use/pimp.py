"""
Delegating package installation to pip, packaging and friends.
"""

import codecs
import importlib.util
import linecache
import os
import platform
import re
import shlex
import sqlite3
import sys
import tarfile
import zipfile
import zipimport
from collections import namedtuple
from collections.abc import Callable
from functools import lru_cache as cache
from functools import reduce
from importlib import metadata
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.metadata import PackageNotFoundError
from itertools import chain
from logging import getLogger
from pathlib import Path, PureWindowsPath, WindowsPath
from pprint import pformat
from subprocess import CalledProcessError, run
from types import ModuleType
from typing import Any, Iterable, Optional, Protocol, TypeVar, Union, runtime_checkable
from warnings import catch_warnings, filterwarnings, warn

import furl
import packaging
import requests
from beartype import beartype
from furl import furl as URL
from icontract import ensure, require
from packaging import tags
from packaging.specifiers import SpecifierSet

import use
from use import Hash, Modes, PkgHash, VersionWarning, config, home
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest
from use.messages import UserMessage, _web_pebkac_no_version_no_hash
from use.pypi_model import PyPI_Project, PyPI_Release, Version, _delete_none
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


def execute_wrapped(sql: str, params: tuple):
    ___use = getattr(sys, "modules").get("use")
    try:
        return getattr(___use, "registry").execute(sql, params)
    except sqlite3.OperationalError as _oe:
        pass
    getattr(___use, "recreate_registry")()
    return getattr(___use, "registry").execute(sql, params)


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
        try:
            from pip._internal.resolution.legacy.resolver import get_supported
        except ImportError:
            pass
        if not get_supported:
            try:
                from pip._internal.models.target_python import get_supported
            except ImportError:
                pass
        if not get_supported:
            try:
                from pip._internal.utils.compatibility_tags import get_supported
            except ImportError:
                pass
        if not get_supported:
            try:
                from pip._internal.resolution.resolvelib.factory import get_supported
            except ImportError:
                pass

    get_supported = get_supported or (lambda: [])

    items: list[PlatformTag] = [PlatformTag(platform=tag.platform) for tag in get_supported()]

    items.extend(PlatformTag(platform=str(tag)) for tag in packaging.tags._platform_tags())

    return frozenset(items)


@beartype
def _filter_by_version(project_: "PyPI_Project", version: Version) -> "PyPI_Project":
    v = Version(version)
    rels = project_.releases.get(v, project_.releases.get(version, []))
    if not rels:
        return project_

    project_.releases = {v: rels}
    project_.urls = rels
    return project_


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
    version: Version = None,
    hash_algo: Hash,
    package_name: str = None,
    no_browser: bool,
    Message: type,
    **kwargs,
) -> Union[Exception, ModuleType]:
    if version is None or version not in _get_package_data_from_pypi(package_name).releases:
        version = next(iter(reversed(_get_package_data_from_pypi(package_name).releases)))
    if hashes := {
        PkgHash(
            "foobar", hexdigest_as_JACK(entry.digests.get(hash_algo.name)), entry.digests.get(hash_algo.name)
        )
        for entry in _get_package_data_from_pypi(package_name).releases[version]
    }:

        proj = _get_package_data_from_pypi(package_name)
        ordered = _filtered_and_ordered_data(proj, version=None)
        recommended_hash = {hexdigest_as_JACK(ordered[0].digests.get(hash_algo.name))}
        return RuntimeWarning(
            Message.pebkac_missing_hash(
                name=name,
                package_name=package_name,
                version=version,
                hashes=hashes,
                recommended_hash=recommended_hash,
                no_browser=no_browser,
            )
        )
    else:
        return RuntimeWarning(Message.no_distribution_found(package_name, version))


@beartype
@pipes
def _pebkac_no_version_no_hash(
    *,
    name: str,
    hash_algo: Hash,
    package_name: str,
    no_browser: bool,
    Message: type,
    **kwargs,
) -> Exception:
    # let's try to make an educated guess and give a useful suggestion
    proj = _get_package_data_from_pypi(package_name)
    ordered = _filtered_and_ordered_data(proj, version=None)
    # we tried our best, but we didn't find anything that could work'
    if not ordered:
        return RuntimeWarning(Message.pebkac_unsupported(package_name))
    # we found something that could work, but it may not fit to the user's requirements
    return RuntimeWarning(
        Message.no_version_or_hash_provided(
            name=name,
            hashes={hexdigest_as_JACK(ordered[0].digests.get(hash_algo.name))},
            package_name=package_name,
            version=ordered[0].version,
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

    # ? does this solve the caching issue?
    if not imported:
        del sys.modules[module_name]
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
def _check_for_installation(*, package_name=str, version) -> Optional[dict]:
    return execute_wrapped(
        """
        SELECT
            artifacts.id, import_relpath,
            artifact_path, installation_path, module_path
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


@beartype
def _auto_install(
    *,
    name: str,
    package_name: str,
    module_name: str,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    version: Version,
    hash_algo: Hash,
    user_provided_hashes: set[int],
    **kwargs,
) -> Union[ModuleType, BaseException]:
    """Install, if necessary, the package and return the module."""
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
        else:
            raise AssertionError(f"{func!r} returned {result!r}")

    query = _check_for_installation(package_name=package_name, version=version)

    if not query or not _ensure_path(query["artifact_path"]).exists():
        query = _find_or_install(package_name=package_name, version=version)

    artifact_path = _ensure_path(query["artifact_path"])
    module_path = _ensure_path(query["module_path"])
    # trying to import directly from zip
    _clean_sys_modules(module_name)
    try:
        importer = zipimport.zipimporter(artifact_path)
        return importer.load_module(query["import_name"])
    except BaseException as zerr:
        pass
    orig_cwd = Path.cwd()
    if "installation_path" not in query or not _ensure_path(query["installation_path"]).exists():
        if query:
            sys.modules["use"].del_entry(name, version)
        query = _find_or_install(package_name=package_name, version=version, force_install=True)
        artifact_path = _ensure_path(query["artifact_path"])
        module_path = _ensure_path(query["module_path"])
    assert "installation_path" in query and query["installation_path"]
    installation_path = _ensure_path(query["installation_path"])
    try:
        module_path = _ensure_path(query["module_path"])
        os.chdir(installation_path)
        return _load_venv_entry(
            module_name=module_name,
            module_path=module_path,
            installation_path=installation_path,
        )

    finally:
        os.chdir(orig_cwd)
        sys.modules["use"]._save_module_info(
            name=package_name,
            import_relpath=str(_ensure_path(module_path).relative_to(installation_path)),
            version=version,
            artifact_path=artifact_path,
            hash_value=hash_algo.value(artifact_path.read_bytes()).hexdigest(),
            module_path=module_path,
            installation_path=installation_path,
        )


def _process(*argv, env={}):
    _realenv = {
        k: v for k, v in chain(os.environ.items(), env.items()) if isinstance(k, str) and isinstance(v, str)
    }
    output = None
    try:
        output = run(
            **(
                setup := dict(
                    executable=argv[0],
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
            )
        )
    except CalledProcessError as err:
        log.error(
            err,
        )
        msg = err  # sic
    return output or msg


@beartype
@ensure(lambda url: str(url).startswith("http"))
def _download_artifact(artifact_path: Path, url: URL) -> bool:
    data = requests.get(url).content
    artifact_path.write_bytes(data)
    return True


@beartype
def _is_pure_python_package(artifact_path: Path, meta: dict) -> bool:
    for n in meta["names"]:
        for s in importlib.machinery.EXTENSION_SUFFIXES:
            if n.endswith(s):
                return False
    if ".tar" in str(artifact_path):
        return False
    return True


@beartype
def _find_module_in_venv(
    *, package_name: str, version: Version, relp: str
) -> tuple[Optional[str], Optional[str]]:
    mod_relative_to_site = None
    try:
        dist = importlib.metadata.Distribution.from_name(package_name)
        for site_dir in (home / "venv" / package_name / str(version)).rglob("**/site-packages"):
            for pp in dist.files:
                relp = "/".join(relp.split("/")[1:])
                if len(relp.split("/")) == 0:
                    break
                if pp.as_posix() == relp:
                    mod_relative_to_site = pp[0]
                    break
    except PackageNotFoundError:
        return None, None
    if mod_relative_to_site is None:
        return None, None

    module_path = site_dir / mod_relative_to_site.as_posix()
    return site_dir, module_path


@beartype
def _find_or_install(
    *,
    package_name: str,
    version: Version = None,
    force_install=False,
) -> dict[str, str]:  # we should make this a proper pydantic class
    """Prepare the installation."""
    out_info = {}

    proj = _get_package_data_from_pypi(package_name)
    if ordered := _filtered_and_ordered_data(proj, version=version):
        release = ordered[0]
    else:
        v = list(proj.releases.keys())[0]
        rel_vals = proj.releases.get(v, [None])
        release = rel_vals[0]
    out_info.update(release.dict())

    url = URL(release.url)
    filename = url.asdict()["path"]["segments"][-1]
    artifact_path = use.home / "packages" / filename

    out_info["artifact_path"] = artifact_path

    if not artifact_path.exists():
        # TODO this will need to be handled - assume the worst!
        _download_artifact(artifact_path, url)

    meta = archive_meta(artifact_path)
    out_info.update(**meta)
    import_parts = re.split("[\\\\/]", meta["import_relpath"])
    if "__init__.py" in import_parts:
        import_parts.remove("__init__.py")
    import_name = ".".join(import_parts)
    relp = meta["import_relpath"]
    out_info["module_path"] = relp
    out_info["import_relpath"] = relp
    out_info["import_name"] = import_name
    if (
        not force_install
        and _is_pure_python_package(artifact_path, meta)
        and str(artifact_path).endswith(".whl")
    ):
        return out_info

    venv_root = home / "venv" / package_name / str(version)
    site_pkgs_dir = list(venv_root.rglob("site-packages"))
    if not any(site_pkgs_dir):
        force_install = True
        module_paths = []
    else:
        out_info["installation_path"] = site_pkgs_dir[0]
        module_paths = venv_root.rglob(f"**/{relp}")

    python_exe = Path(sys.executable)

    if not module_paths or force_install:
        # If we get here, the venv/pip setup is required.
        output = _process(
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
        )
    installation_path, module_path = _find_module_in_venv(
        package_name=package_name, version=version, relp=relp
    )
    out_info.update(
        {
            "artifact_path": artifact_path,
            "installation_path": installation_path,
            "module_path": module_path,
            "import_relpath": ".".join(relp.split("/")[:-1]),
        }
    )
    return _delete_none(out_info)


@beartype
def _load_venv_entry(*, module_name: str, installation_path: Path, module_path: Path) -> ModuleType:
    cwd = Path.cwd()
    orig_exc = None
    old_sys_path = list(sys.path)
    if sys.path[0] != "":
        sys.path.insert(0, "")
    try:
        for variant in (
            cwd,
            installation_path,
            Path(str(installation_path).replace("lib64/", "lib/")),
            Path(str(installation_path).replace("lib/", "lib64/")),
        ):
            if not variant.exists():
                continue
            origcwd = Path.cwd()
            try:
                os.chdir(variant)
                return importlib.import_module(module_name)
            except ImportError as ierr0:
                orig_exc = orig_exc or ierr0
                continue
            finally:
                os.chdir(origcwd)
    except RuntimeError as ierr:
        try:
            return importlib.import_module(module_name)
        except BaseException as ierr2:
            raise ierr from orig_exc
    finally:
        os.chdir(cwd)
        sys.path = old_sys_path


@cache(maxsize=512)
def _get_package_data_from_pypi(package_name: str) -> PyPI_Project:
    json_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(json_url)
    if response.status_code == 404:
        raise ImportError(UserMessage.pebkac_unsupported(package_name))
    elif response.status_code != 200:
        raise RuntimeWarning(UserMessage.web_error(json_url, response))
    return PyPI_Project(**response.json())


@beartype
def _filter_by_platform(
    project: PyPI_Project, tags: frozenset[PlatformTag], sys_version: Version
) -> PyPI_Project:
    filtered = {
        ver: [
            rel.dict()
            for rel in releases
            if _is_compatible(
                rel,
                sys_version=sys_version,
                platform_tags=tags,
                include_sdist=False,
            )
        ]
        for ver, releases in project.releases.items()
    }
    if not filtered:
        filtered = {
            ver: [
                rel.dict()
                for rel in releases
                if _is_compatible(
                    rel,
                    sys_version=sys_version,
                    platform_tags=tags,
                    include_sdist=True,
                )
            ]
            for ver, releases in project.releases.items()
        }

    return PyPI_Project(**{**project.dict(), **{"releases": filtered}})


@beartype
@pipes
def _filtered_and_ordered_data(data: PyPI_Project, version: Optional[Version] = None) -> list[PyPI_Release]:
    sys_version = Version(major=sys.version_info[0], minor=sys.version_info[1])
    if version:
        filtered = (
            data
            >> _filter_by_version(version)
            >> _filter_by_platform(tags=get_supported(), sys_version=sys_version)
        )
    else:
        filtered = data
        filtered = _filter_by_platform(data, tags=get_supported(), sys_version=sys_version)

    flat = reduce(list.__add__, filtered.releases.values(), [])
    return sorted(
        flat,
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
    supported_tags = set(
        [
            our_python_tag,
            "py3",
            "cp3",
            f"cp{tags.interpreter_version()}",
            f"py{tags.interpreter_version()}",
        ]
    )

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


def _is_compatible(info: PyPI_Release, sys_version, platform_tags, include_sdist=None) -> bool:
    """Return true if the artifact described by 'info'
    is compatible with the current or specified system."""
    specifier = info.requires_python

    return (
        _is_platform_compatible(info, platform_tags, include_sdist)
        and not info.yanked
        and (include_sdist or info.justuse.ext not in ("tar", "tar.gz" "zip"))
    )


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
        version = version.__call__()
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
        return default  # TODO: write test for default
    else:
        raise exception
