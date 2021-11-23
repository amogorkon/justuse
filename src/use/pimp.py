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
from functools import lru_cache as cache
from functools import reduce
from importlib import metadata
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.metadata import PackageNotFoundError
from itertools import chain
from logging import getLogger
from pathlib import Path, PureWindowsPath, WindowsPath
from pprint import pformat
from subprocess import run
from types import ModuleType
from typing import Any, Callable, Optional, Union
from warnings import warn

import furl
import packaging
import requests
from furl import furl as URL
from icontract import ensure, require
from packaging import tags
from packaging.specifiers import SpecifierSet
from pip._internal.utils import compatibility_tags

import use
from use import Hash, Modes, VersionWarning, config
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest
from use.messages import Message, _web_no_version_or_hash_provided
from use.pypi_model import PyPI_Project, PyPI_Release, Version, _delete_none
from use.tools import pipes

log = getLogger(__name__)


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


def _ensure_version(
    result: Union[ModuleType, Exception], *, name, version, **kwargs
) -> Union[ModuleType, Exception]:
    if not isinstance(result, ModuleType):
        return result
    result_version = _get_version(mod=result)
    if result_version != version:
        warn(Message.version_warning(name, version, result_version), category=VersionWarning)
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
def get_supported() -> frozenset[PlatformTag]:
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, package_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    items: list[PlatformTag] = []

    for tag in compatibility_tags.get_supported():
        items.append(PlatformTag(platform=tag.platform))
    for tag in packaging.tags._platform_tags():
        items.append(PlatformTag(platform=str(tag)))

    return frozenset(items)


def _filter_by_version(project_: "PyPI_Project", version: str) -> "PyPI_Project":

    for_version = (
        project_.releases.get(version)
        or project_.releases.get(str(version))
        or project_.releases.get(Version(str(version)))
    )
    new_data = {
        "urls": for_version,
        "releases": {Version(str(version)): for_version},
        "info": project_.info.dict(),
    }
    return PyPI_Project(**new_data)


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


@pipes
def archive_meta(artifact_path):
    DIST_PKG_INFO_REGEX = re.compile("(dist-info|-INFO|\\.txt$|(^|/)[A-Z0-9_-]+)$")
    meta = archive = names = functions = None

    if ".tar" in Path(str(artifact_path)).stem:
        functions = TarFunctions(artifact_path)
    else:
        functions = ZipFunctions(artifact_path)

    archive, names = functions.get()
    meta = dict(names << filter(DIST_PKG_INFO_REGEX.search) << map(functions.read_entry))
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
    archive.close()
    return meta


def _clean_sys_modules(package_name: str) -> None:
    if not package_name:
        return
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


def _venv_root(package_name, version, home) -> Path:
    assert version
    return home / "venv" / package_name / str(version)


def _pebkac_no_version(
    *,
    name: str,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    version: Version = None,
    hash_algo=None,
    package_name: str = None,
    module_name: str = None,
    message_formatter: Callable[[str, str, Version, set[str]], str] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[ModuleType, Exception]:

    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result

    return RuntimeWarning(Message.cant_import_no_version(name))


def _pebkac_no_hash(
    *,
    version: Version = None,
    hash_algo: Hash,
    package_name: str = None,
    **kwargs,
) -> Union[Exception, ModuleType]:
    hashes = {
        hexdigest_as_JACK(entry.digests.get(hash_algo.name))
        for entry in _get_package_data(package_name).releases[version]
    }
    if hashes:
        return RuntimeWarning(Message.pebkac_missing_hash(hash_algo.name, hashes))
    else:
        return RuntimeWarning(Message.no_distribution_found(package_name, version))


@pipes
def _pebkac_no_version_no_hash(
    *,
    name: str,
    hash_algo: Hash,
    package_name: str,
    **kwargs,
) -> Exception:
    # let's try to make an educated guess and give a useful suggestion
    proj = _get_package_data(package_name)
    ordered = _filtered_and_ordered_data(proj, version=None)
    # we tried our best, but we didn't find anything that could work'
    if not ordered:
        return RuntimeWarning(Message.pebkac_unsupported(package_name))

    # we found something that could work, but it may not fit to the user's requirements
    hashes = {hexdigest_as_JACK(o.digests.get(hash_algo.name)) for o in proj.urls}
    version = ordered[0].version
    return RuntimeWarning(
        Message.no_version_or_hash_provided(
            name=name,
            hashes=hashes,
            package_name=package_name,
            version=version,
        )
    )


def _import_public_no_install(
    *,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    package_name: str = None,
    module_name: str = None,
    spec=None,
    **kwargs,
) -> Union[Exception, ModuleType]:
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
    # builtin?
    builtin = False
    try:
        metadata.PathDistribution.from_name(package_name)
    except metadata.PackageNotFoundError:  # indeed builtin!
        builtin = True

    if builtin:
        if spec.name in sys.modules:
            mod = sys.modules[spec.name]
        if mod is None:
            mod = importlib.import_module(module_name)
        assert mod
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # ! => cache
        return mod
    # it seems to be installed in some way, for instance via pip
    return importlib.import_module(module_name)  # ! => cache


def _parse_name(name) -> tuple[str, str]:
    match = re.match(r"(?P<package_name>[^/.]+)/?(?P<rest>[a-zA-Z0-9._]+)?$", name)
    assert match, f"Invalid name spec: {name!r}"
    names = match.groupdict()
    package_name = names["package_name"]
    rest = names["rest"]
    if not package_name:
        package_name = rest
    if not rest:
        rest = package_name
    return (package_name, rest)


def _auto_install(
    *,
    name: str,
    func: Callable[..., Union[Exception, ModuleType]] = None,
    version: Version,
    hash_algo,
    package_name: str,
    module_name: str,
    **kwargs,
) -> Union[ModuleType, BaseException]:
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result

    query = execute_wrapped(
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

    if not query or not _ensure_path(query["artifact_path"]).exists():
        query = _find_or_install(package_name, version)

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
    mod = None
    if "installation_path" not in query or not _ensure_path(query["installation_path"]).exists():
        if query:
            use.del_entry(name, version)
        query = _find_or_install(package_name, version, force_install=True)
        artifact_path = _ensure_path(query["artifact_path"])
        module_path = _ensure_path(query["module_path"])
    assert "installation_path" in query  # why redundant assertions?
    assert query["installation_path"]
    installation_path = _ensure_path(query["installation_path"])
    try:
        module_path = _ensure_path(query["module_path"])
        os.chdir(installation_path)
        import_name = (
            str(module_path.relative_to(installation_path))
            .replace("\\", "/")
            .replace("/__init__.py", "")
            .replace("-", "_")
        )
        return (
            mod := _load_venv_entry(
                package_name=package_name,
                module_name=module_name,
                module_path=module_path,
                installation_path=installation_path,
            )
        )

    finally:
        os.chdir(orig_cwd)
        if "fault_inject" in config:
            config["fault_inject"](**locals())
        if mod:
            use.main._save_module_info(
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
    o = run(
        **(
            setup := dict(
                executable=(exe := argv[0]),
                args=[*map(str, argv)],
                bufsize=1024,
                input="",
                capture_output=True,
                timeout=45000,
                check=False,
                close_fds=True,
                env=_realenv,
                encoding="ISO-8859-1",
                errors="ISO-8859-1",
                text=True,
                shell=False,
            )
        )
    )
    if o.returncode == 0:
        return o
    raise RuntimeError(  # cov: exclude
        "\x0a".join(
            (
                "\x1b[1;41;37m",
                f"Problem running--command exited with non-zero: {o.returncode}",
                f"{shlex.join(map(str, setup['args']))}",
                "---[  Errors  ]---",
                f"{o.stderr or o.stdout}",
                "\x1b[0;1;37m",
                "Arguments to subprocess.run(**setup):",
                f"{pformat(setup, indent=2, width=70, compact=False)}",
                "---[  STDOUT  ]---",
                f"{o.stdout}",
                "---[  STDERR  ]---",
                f"{o.stderr}\x1b[0m",
            )
        )
        if o.returncode != 0
        else (f"{o.stdout}\n\n{o.stderr}")
    )


def _get_venv_env(venv_root: Path) -> dict[str, str]:
    pathvar = os.environ.get("PATH")
    python_exe = Path(sys.executable)
    if not venv_root.exists():
        venv_root.mkdir(parents=True)
    exe_dir = python_exe.parent.absolute()
    return {}


@ensure(lambda url: str(url).startswith("http"))  # should we be enforcing 443 > 80?
def _download_artifact(name, version, filename, url) -> Path:
    artifact_path = (sys.modules["use"].home / "packages" / filename).absolute()
    if not artifact_path.exists():
        data = requests.get(url).content
        artifact_path.write_bytes(data)
    return artifact_path


def _pure_python_package(artifact_path, meta):
    not_pure_python = any(
        any(n.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES) for n in meta["names"]
    )

    if ".tar" in str(artifact_path):
        return False
    if not_pure_python:
        return False
    return True


def _find_module_in_venv(package_name, version, relp):
    ___use = getattr(sys, "modules").get("use")
    ret = None, None
    site_dirs = list(
        (getattr(___use, "home") / "venv" / package_name / str(version)).rglob("**/site-packages")
    )
    if not site_dirs:
        return ret

    dist = None
    mod_relative_to_site = None
    osp = None
    for site_dir in site_dirs:
        osp = list(sys.path)
        sys.path.clear()
        sys.path.insert(0, str(site_dir))
        try:
            dist = importlib.metadata.Distribution.from_name(package_name)
            while not mod_relative_to_site:
                pps = [pp for pp in dist.files if pp.as_posix() == relp]
                if pps:
                    mod_relative_to_site = pps[0]
                    break
                if len(relp.split("/")) == 0:
                    break
                relp = "/".join(relp.split("/")[1:])
        except PackageNotFoundError:
            continue
        finally:
            sys.path.remove(str(site_dir))
            sys.path += osp
    if mod_relative_to_site:
        module_path = site_dir / mod_relative_to_site.as_posix()
        return (ret := site_dir, module_path)
    return ret


def _find_or_install(name, version=None, artifact_path=None, url=None, out_info=None, force_install=False):
    if out_info is None:
        out_info = {}
    info = out_info
    package_name, rest = _parse_name(name)

    if isinstance(url, str):
        url = URL(url)
    filename = artifact_path.name if artifact_path else None
    if url:
        filename = url.asdict()["path"]["segments"][-1]
    else:
        filename = artifact_path.name if artifact_path else None
    if filename and not artifact_path:
        artifact_path = use.home / "packages" / filename

    if not url or not artifact_path or (artifact_path and not artifact_path.exists()):
        proj = _get_package_data(package_name)
        ordered = _filtered_and_ordered_data(proj, version=version)
        rel = None
        for r in ordered:
            rel = r
            break
        if not rel:
            v = list(proj.releases.keys())[0]
            rel_vals = proj.releases.get(v, [None])
            rel = rel_vals[0]
        info.update(rel.dict())
        out_info.update(rel.dict())

        url = URL(rel.url)
        filename = url.asdict()["path"]["segments"][-1]
        artifact_path = use.home / "packages" / filename

    out_info["artifact_path"] = artifact_path
    # todo: set info
    as_dict = info
    url = URL(as_dict["url"])
    filename = url.path.segments[-1]
    info["filename"] = filename
    # info.update(_parse_filename(filename))
    info = {**info, "version": Version(version)}
    if not artifact_path.exists():
        artifact_path = _ensure_path(_download_artifact(name, version, filename, url))

    out_info.update(info)
    install_item = artifact_path
    meta = archive_meta(artifact_path)
    import_parts = re.split("[\\\\/]", meta["import_relpath"])
    if "__init__.py" in import_parts:
        import_parts.remove("__init__.py")
    import_name = ".".join(import_parts)
    name = f"{package_name}.{import_name}"
    relp = meta["import_relpath"]
    out_info["module_path"] = relp
    out_info["import_relpath"] = relp
    out_info["import_name"] = import_name
    if (
        not force_install
        and _pure_python_package(artifact_path, meta)
        and str(artifact_path).endswith(".whl")
    ):
        return out_info

    venv_root = _venv_root(package_name, version, use.home)
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
            install_item,
        )
        sys.stderr.write("\n\n".join((output.stderr, output.stdout)))

    site_dir, module_path = _find_module_in_venv(package_name, version, relp)
    module_paths = []
    if module_path:
        module_paths.append(module_path)
        installation_path = site_dir
        site_pkgs_dir = site_dir

    out_info.update(**meta)
    assert module_paths
    out_info.update(
        {
            **info,
            "artifact_path": artifact_path,
            "installation_path": installation_path,
            "module_path": module_path,
            "import_relpath": ".".join(relp.split("/")[0:-1]),
            "info": info,
        }
    )
    return _delete_none(out_info)


def _load_venv_entry(*, package_name, module_name, installation_path, module_path) -> ModuleType:
    cwd = Path.cwd()
    orig_exc = None
    old_sys_path = list(sys.path)
    if sys.path[0] != "":
        sys.path.insert(0, "")
    with open(module_path, "rb") as code_file:
        try:
            for variant in (
                installation_path,
                Path(str(str(installation_path).replace("lib64/", "lib/"))),
                Path(str(str(installation_path).replace("lib/", "lib64/"))),
                None,
            ):
                if not variant:
                    raise RuntimeError()
                if not variant.exists():
                    continue
                try:
                    os.chdir(cwd)
                    os.chdir(variant)
                    return _build_mod(
                        name=(module_name.replace("/", ".")),
                        code=code_file.read(),
                        module_path=_ensure_path(module_path),
                        initial_globals={},
                    )
                except ImportError as ierr0:
                    orig_exc = orig_exc or ierr0
                    continue
        except RuntimeError as ierr:
            try:
                return importlib.import_module(module_name)
            except BaseException as ierr2:
                raise ierr from orig_exc
        finally:
            os.chdir(cwd)
            sys.path.clear()
            for p in old_sys_path:
                sys.path.append(p)


@cache(maxsize=512)
def _get_package_data(package_name: str) -> PyPI_Project:
    json_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(json_url)
    if response.status_code == 404:
        raise ImportError(Message.pebkac_unsupported(package_name))
    elif response.status_code != 200:
        raise RuntimeWarning(Message.web_error(json_url, response))
    return PyPI_Project(**response.json())


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
                include_sdist=True,
            )
        ]
        for ver, releases in project.releases.items()
    }

    return PyPI_Project(**{**project.dict(), **{"releases": filtered}})


@pipes
def _filtered_and_ordered_data(data: PyPI_Project, version: Version = None) -> list[PyPI_Release]:
    if version:
        version = Version(str(version))
        filtered = (
            data
            >> _filter_by_version(version)
            # >> _filter_by_platform(tags=get_supported(), sys_version=_sys_version())  # let's not filter by platform for now
        )
    else:
        filtered = data
        # filtered = _filter_by_platform(data, tags=get_supported(), sys_version=_sys_version())

    flat = reduce(list.__add__, filtered.releases.values(), [])
    return sorted(
        flat,
        key=lambda r: (not r.filename.endswith(".tar.gz"), not r.is_sdist, r.version),
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


@pipes
def _is_platform_compatible(
    info: PyPI_Release, platform_tags: frozenset[PlatformTag], include_sdist=False
) -> bool:

    if "py2" in info.justuse.python_tag and "py3" not in info.justuse.python_tag:
        return False

    if not include_sdist and (".tar" in info.justuse.ext or info.justuse.python_tag in ("cpsource", "sdist")):
        return False

    if "win" in info.packagetype and sys.platform != "win32":
        return False

    if "win32" in info.justuse.platform_tag and sys.platform != "win32":
        return False

    if "macosx" in info.justuse.platform_tag and sys.platform != "darwin":
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

    given_platform_tags = info.justuse.platform_tag.split(".") << map(PlatformTag) >> frozenset

    if info.is_sdist and info.requires_python is not None:
        given_python_tag = {
            our_python_tag
            for p in info.requires_python.split(",")
            if Version(platform.python_version()) in SpecifierSet(p)
        }
    else:
        given_python_tag = set(info.justuse.python_tag.split("."))
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


def _apply_aspect(
    thing,
    check,
    pattern,
    decorator: Callable[[Callable[..., Any]], Any],
    aspectize_dunders=False,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    for name, obj in thing.__dict__.items():
        if not aspectize_dunders and name.startswith("__") and name.endswith("__"):
            continue
        if check(obj) and re.match(pattern, name):
            thing.__dict__[name] = decorator(obj)
            log.debug(f"Applied {decorator.__name__} to {obj.__qualname__}")
    return thing


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
    name,  # TODO: this should be a package name and module name
    code,
    initial_globals: Optional[dict[str, Any]],
    module_path,
) -> ModuleType:

    package_name, rest = _parse_name(name)
    mod = ModuleType(rest)

    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__path__ = [str(module_path.parent)]
    mod.__package__ = package_name
    mod.__name__ = rest
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
