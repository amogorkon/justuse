import codecs
import importlib.util
import inspect
import linecache
import os
import platform
import re
import requests
import shlex
import sys
import tarfile
import zipfile
import zipimport
from enum import Enum
from functools import lru_cache as cache
from functools import reduce
from importlib import metadata
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.metadata import PackageNotFoundError
from importlib.util import find_spec
from itertools import chain
from pathlib import PureWindowsPath, WindowsPath
from pprint import pformat
from subprocess import run
from types import ModuleType
from typing import Any, Callable, Optional, Union
from pathlib import Path

import furl
import packaging
from furl import furl as URL
from icontract import ensure
from packaging import tags
from pip._internal.utils import compatibility_tags
from packaging.specifiers import SpecifierSet

from .. import hash_alphabet
from . import Decorators as D
from .Hashish import Hash
from .init_conf import config, use
from .Messages import AmbiguityWarning, Message
from .PlatformTag import PlatformTag
from .init_conf import log
from ..pypi_model import PyPI_Release

from ..pypi_model import PyPI_Project, Version

mode = Enum("Mode", "fastfail")


def all_kwargs(func, other_locals):
    d = {
        name: other_locals[name]
        for name, param in inspect.signature(func).parameters.items()
        if (
            param.kind is inspect.Parameter.KEYWORD_ONLY
            or param.kind is inspect.Parameter.VAR_KEYWORD
        )
    }
    d.update(d["kwargs"])
    del d["kwargs"]
    return d


@D.pipes
def _ensure_path(value: Union[bytes, str, furl.Path, Path]) -> Path:
    if isinstance(value, (str, bytes)):
        return Path(value).absolute()
    if isinstance(value, furl.Path):
        return (
            Path.cwd(),
            value.segments << map(Path) << tuple << reduce(Path.__truediv__),
        ) << reduce(Path.__truediv__)
    return value


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
    log.debug("enter get_supported()")
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

    tags = frozenset(items)
    log.debug("leave get_supported() -> %s", repr(tags))
    return tags


def sort_releases_by_install_method(project: PyPI_Project) -> PyPI_Project:
    return PyPI_Project(
        **{
            **project.dict(),
            **{
                "releases": {
                    k: [x.dict() for x in sorted(v, key=lambda r: r.is_sdist)]
                    for k, v in project.releases.items()
                }
            },
        }
    )


def recommend_best_version(project: PyPI_Project) -> list:
    sorted(project.releases.keys(), reverse=True)


def _filter_by_version(project: PyPI_Project, version: str) -> PyPI_Project:
    return PyPI_Project(
        **{
            **project.dict(),
            **{
                "releases": {version: [v.dict() for v in project.releases[version]]}
                if project.releases.get(version)
                else {}
            },
        }
    )


class TarFunctions:
    def __init__(self, artifact_path):
        self.archive = tarfile.open(artifact_path)

    def get(self):
        return (
            self.archive,
            [m.name for m in self.archive.getmembers() if m.type == b"0"],
        )

    def read_entry(self, entry_name):
        m = self.archive.getmember(entry_name)
        with self.archive.extractfile(m) as f:
            bdata = f.read()
            text = bdata.decode("ISO-8859-1") if len(bdata) < 8192 else ""
            return (Path(entry_name).stem, text.splitlines())


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


@D.pipes
def archive_meta(artifact_path):
    DIST_PKG_INFO_REGEX = re.compile("(dist-info|-INFO|\\.txt$|(^|/)[A-Z0-9_-]+)$")
    meta = archive = names = functions = None

    if ".tar" in str(artifact_path):
        archive = tarfile.open(artifact_path)

        def get_archive(artifact_path):
            archive = tarfile.open(artifact_path)
            return (archive, [m.name for m in archive.getmembers() if m.type == b"0"])

        functions = TarFunctions(artifact_path)
    else:

        def get_archive(artifact_path):
            archive = zipfile.ZipFile(artifact_path)
            return (archive, [e.filename for e in archive.filelist])

        functions = ZipFunctions(artifact_path)

    archive, names = get_archive(artifact_path)
    archive, names = functions.get()
    meta = dict(
        names << filter(DIST_PKG_INFO_REGEX.search) << map(functions.read_entry)
    )
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
    import_name = (name,) if (top_level == name) else (top_level, name)
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


def _ensure_loader(obj: Union[ModuleType, ModuleSpec]):
    loader = None
    if not loader and isinstance(obj, ModuleType):
        loader = obj.__loader__
    if (
        not loader
        and isinstance(obj, ModuleType)
        and (spec := getattr(obj, "__spec__", None))
    ):
        loader = spec.loader
    if not loader and isinstance(obj, ModuleSpec):
        loader = obj.loader
    if not loader and hasattr(importlib.util, "loader_from_spec"):
        loader = importlib.util.loader_from_spec(obj)
    if not loader and isinstance(obj, ModuleType):
        name = obj.__name__
        mod = obj
        segments = name.split(".")
        parent_mod = importlib.import_module(".".join(segments[:-1]))
        parent_spec = parent_mod.__spec__
        ctor_args = [
            (
                k,
                getattr(mod, "__name__")
                if "name" in k
                else (
                    list(
                        Path(parent_spec.submodule_search_locations[0]).glob(
                            mod.__name__[len(parent_mod.__name__) + 1 :] + ".*"
                        )
                    )
                    + list(
                        Path(parent_spec.submodule_search_locations[0]).glob(
                            mod.__name__[len(parent_mod.__name__) + 1 :]
                            + "/__init__.py"
                        )
                    )
                )[0]
                if ("path" in k or "file" in k or "loc" in k)
                else k,
            )
            for k in list(
                inspect.signature(type(parent_mod.__loader__).__init__).parameters
            )[1:]
        ]
        loader = type(parent_mod.__loader__)(*ctor_args)
    if not loader:
        if isinstance(obj, ModuleType):
            name = obj.__name__
            spec = obj.__spec__
        if isinstance(obj, ModuleSpec):
            name = obj.name
            spec = obj
        module_path = (
            getattr(spec, "origin", None)
            or getattr(obj, "__file__", None)
            or getattr(obj, "__path__", None)
            or inspect.getsourcefile(obj)
        )
        loader = SourceFileLoader(name, module_path)
    return loader


def _clean_sys_modules(package_name: str) -> None:
    for k in dict(
        [
            (k, _ensure_loader(v))
            for k, v in sys.modules.items()
            if (
                getattr(v, "__spec__", None) is None
                or isinstance(
                    _ensure_loader(v), (SourceFileLoader, zipimport.zipimporter)
                )
            )
            and package_name in k.split(".")
        ]
    ):
        if k in sys.modules:
            del sys.modules[k]


def _venv_root(package_name, version, home) -> Path:
    assert version
    return home / "venv" / package_name / str(version)


def _pebkac_no_version_hash(
    *,
    name: str,
    func: Callable[[...], Union[Exception, ModuleType]]=None,
    version: Version=None,
    hash_algo=None,
    package_name: str=None,
    module_name: str=None,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[ModuleType, Exception]:

    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
    
    return RuntimeWarning(Message.cant_import_no_version(name))


def _pebkac_version_no_hash(
    *,
    name: str,
    func: Callable[[...], Union[Exception, ModuleType]]=None,
    version: Version=None,
    hash_algo=None,
    package_name: str=None,
    module_name: str=None,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[Exception, ModuleType]:
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
    try:
        hashes = {
            hash_alphabet.hexdigest_as_JACK(entry.digests.get(hash_algo.name))
            for entry in (_get_package_data(package_name).releases[version])
        }
        if not hashes:
            rw = RuntimeWarning(Message.pebkac_unsupported(package_name))
        else:
            rw = RuntimeWarning(message_formatter(name, package_name, version, hashes))
        rw.name = name
        rw.version = version
        rw.hashes = hashes
        return rw
    except (IndexError, KeyError) as ike:
        return RuntimeWarning(Message.no_distribution_found(package_name, version))


@D.pipes
def _pebkac_no_version_no_hash(
    *,
    name: str,
    func: Callable[[...], Union[Exception, ModuleType]]=None,
    version: Version=None,
    hash_algo=None,
    package_name: str=None,
    module_name: str=None,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[Exception, ModuleType]:
    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
    # let's try to make an educated guess and give a useful suggestion
    data = _get_package_data(package_name) >> _filter_by_platform(
        tags=get_supported(), sys_version=_sys_version()
    )
    flat = reduce(list.__add__, data.releases.values(), [])
    priority = sorted(flat, key=lambda r: (not r.is_sdist, r.version), reverse=True)

    for info in priority:
        return _pebkac_version_no_hash(
            func=None,
            name=name,
            version=info.version,
            hash_algo=hash_algo,
            package_name=package_name,
            message_formatter=Message.no_version_or_hash_provided,
        )

    rw = RuntimeWarning(Message.pebkac_unsupported(package_name))
    rw.name = package_name
    return rw


def _import_public_no_install(
    *,
    name: str,
    func: Callable[[...], Union[Exception, ModuleType]]=None,
    version: Version=None,
    hash_algo=None,
    package_name: str=None,
    module_name: str=None,
    spec=None,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
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
        return _extracted_from__import_public_no_install_18(module_name, spec)
    # it seems to be installed in some way, for instance via pip
    return importlib.import_module(module_name)  # ! => cache


# TODO Rename this here and in `_import_public_no_install`
def _extracted_from__import_public_no_install_18(module_name, spec):
    if spec.name in sys.modules:
        mod = sys.modules[spec.name]
        importlib.reload(mod)
    else:
        mod = _ensure_loader(spec).create_module(spec)
    if mod is None:
        mod = importlib.import_module(module_name)
    assert mod
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # ! => cache
    return mod


def _parse_name(name) -> tuple[str, str]:
    ret = name, name
    try:
        match = re.match(r"(?P<package_name>[^/.]+)/?(?P<rest>[a-zA-Z0-9._]+)?$", name)
        assert match, f"Invalid name spec: {name!r}"
        names = match.groupdict()
        package_name = names["package_name"]
        rest = names["rest"]
        if not package_name:
            package_name = rest
        if not rest:
            rest = package_name
        ret = (package_name, rest)
        return ret
    finally:
        log.info("_parse_name(%s) -> %s", repr(name), repr(ret))


def _auto_install(
    *,
    name: str,
    func: Callable[[...], Union[Exception, ModuleType]]=None,
    version: Version=None,
    hash_algo=None,
    package_name: str=None,
    module_name: str=None,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[ModuleType, BaseException]:
    package_name, rest = _parse_name(name)

    if func:
        result = func()
        if isinstance(result, (Exception, ModuleType)):
            return result
    
    query = execute_wrapped(
        f"""
        SELECT
            artifacts.id, import_relpath,
            artifact_path, installation_path, module_path
        FROM distributions
        JOIN artifacts ON artifacts.id = distributions.id
        WHERE name=? AND version=?
        ORDER BY artifacts.id DESC
        """,
        (package_name, str(version), ),
    ).fetchone()

    if not query or not _ensure_path(query["artifact_path"]).exists():
        query = _find_or_install(package_name, version)

    artifact_path = _ensure_path(query["artifact_path"])
    module_path = _ensure_path(query["module_path"])
    # trying to import directly from zip
    _clean_sys_modules(rest)
    try:
        importer = zipimport.zipimporter(artifact_path)
        return importer.load_module(query["import_name"])
    except BaseException as zerr:
        pass
    orig_cwd = Path.cwd()
    mod = None
    if "installation_path" not in query:
        query = _find_or_install(package_name, version, force_install=True)
        artifact_path = _ensure_path(query["artifact_path"])
        module_path = _ensure_path(query["module_path"])
    assert "installation_path" in query
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
                package_name,
                import_name,
                module_path=module_path,
                installation_path=installation_path,
            )
        )

    finally:
        os.chdir(orig_cwd)
        if "fault_inject" in config:
            config["fault_inject"](**locals())
        if mod:
            use._save_module_info(
                name=package_name,
                import_relpath=str(
                    _ensure_path(module_path).relative_to(installation_path)
                ),
                version=version,
                artifact_path=artifact_path,
                hash_value=hash_algo.value(artifact_path.read_bytes()).hexdigest(),
                module_path=module_path,
                installation_path=installation_path,
            )


def _process(*argv, env={}):
    _realenv = {
        k: v
        for k, v in chain(os.environ.items(), env.items())
        if isinstance(k, str) and isinstance(v, str)
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
    raise RuntimeError(
        "\x0a".join(
            (
                "\x1b[1;41;37m",
                "Problem running--command exited with non-zero: %d",
                "%s",
                "---[  Errors  ]---",
                "%s",
                "\x1b[0;1;37m",
                "Arguments to subprocess.run(**setup):",
                "%s",
                "---[  STDOUT  ]---",
                "%s",
                "---[  STDERR  ]---",
                "%s\x1b[0m",
            )
        )
        % (
            o.returncode,
            shlex.join(map(str, setup["args"])),
            o.stderr or o.stdout,
            pformat(setup, indent=2, width=70, compact=False),
            o.stdout,
            o.stderr,
        )
        if o.returncode != 0
        else ("%s\n\n%s") % (o.stdout, o.stderr)
    )


def _find_version(package_name, version=None) -> PyPI_Release:
    data = _filtered_and_ordered_data(_get_package_data(package_name), version)
    return data[0]


def _get_venv_env(venv_root: Path) -> dict[str, str]:
    pathvar = os.environ.get("PATH")
    python_exe = Path(sys.executable)
    if not venv_root.exists():
        venv_root.mkdir(parents=True)
    exe_dir = python_exe.parent.absolute()
    return {}


@ensure(lambda url: str(url).startswith("http"))
def _download_artifact(name, version, filename, url) -> Path:
    artifact_path = (sys.modules["use"].home / "packages" / filename).absolute()
    if not artifact_path.exists():
        log.info("Downloading %s==%s from %s", name, version, url)
        data = requests.get(url).content
        artifact_path.write_bytes(data)
        log.debug("Wrote %d bytes to %s", len(data), artifact_path)
    return artifact_path


def _delete_none(a_dict: dict[str, object]) -> dict[str, object]:
    for k, v in tuple(a_dict.items()):
        if v is None or v == "":
            del a_dict[k]
    return a_dict


def _pure_python_package(artifact_path, meta):
    not_pure_python = any(
        any(n.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES)
        for n in meta["names"]
    )

    if ".tar" in str(artifact_path):
        return False
    if not_pure_python:
        return False
    return True


def _find_module_in_venv(package_name, version, relp):
    ___use = getattr(sys, "modules").get("use")
    ret = None, None
    try:
        site_dirs = list(
            (
                getattr(___use, "home") 
                / "venv" / package_name 
                / str(version)
            ).rglob("**/site-packages")
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
    finally:
        log.info(
            "_find_module_in_venv(package_name=%s, version=%s, relp=%s) -> %s",
            repr(package_name),
            repr(version),
            repr(relp),
            repr(ret),
        )


def _find_or_install(
    name, version=None, artifact_path=None, url=None, out_info=None, force_install=False
):
    log.debug(
        "_find_or_install(name=%s, version=%s, artifact_path=%s, url=%s)",
        name,
        version,
        artifact_path,
        url,
    )
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
        artifact_path = sys.modules["use"].home / "packages" / filename

    if not url or not artifact_path or (artifact_path and not artifact_path.exists()):
        info.update(
            _filtered_and_ordered_data(_get_package_data(package_name), version)[
                0
            ].dict()
        )
        url = URL(str(info["url"]))
        filename = url.asdict()["path"]["segments"][-1]
        artifact_path = sys.modules["use"].home / "packages" / filename
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
    if not force_install and _pure_python_package(artifact_path, meta):
        return out_info

    venv_root = _venv_root(package_name, version, use.home)
    out_info["installation_path"] = venv_root
    python_exe = Path(sys.executable)
    env = _get_venv_env(venv_root)
    module_paths = venv_root.rglob(f"**/{relp}")
    if force_install or (not python_exe.exists() or not any(module_paths)):
        log.info("calling pip to install install_item=%s", install_item)

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

    out_info.update(**meta)
    assert module_paths

    log.info("installation_path = %s", installation_path)
    log.info("module_path = %s", module_path)
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


def _load_venv_entry(package_name, rest, installation_path, module_path) -> ModuleType:
    log.info(
        "load_venv_entry package_name=%s rest=%s module_path=%s",
        package_name,
        rest,
        module_path,
    )
    cwd = Path.cwd()
    log.info(f"{cwd=}")
    log.info(f"{sys.path=}")
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
                        name=(package_name + "/" + rest.replace("/", ".")),
                        code=code_file.read(),
                        module_path=_ensure_path(module_path),
                        initial_globals={},
                    )
                except ImportError as ierr0:
                    orig_exc = orig_exc or ierr0
                    continue
        except RuntimeError as ierr:
            try:
                return importlib.import_module(rest)
            except BaseException as ierr2:
                raise ierr from orig_exc
        finally:
            os.chdir(cwd)
            sys.path.clear()
            for p in old_sys_path:
                sys.path.append(p)


@cache(maxsize=512)
def _get_package_data(package_name) -> PyPI_Project:
    json_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(json_url)
    if response.status_code == 404:
        raise ImportError(Message.pebkac_unsupported(package_name))
    elif response.status_code != 200:
        raise RuntimeWarning(Message.web_error(json_url, response))
    return PyPI_Project(**response.json())


def _sys_version():
    return Version(".".join(map(str, sys.version_info[0:3])))


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


@D.pipes
def _filtered_and_ordered_data(
    data: PyPI_Project, version: Version = None
) -> list[PyPI_Release]:
    if version:
        version = Version(str(version))
        filtered = (
            data
            >> _filter_by_version(version)
            >> _filter_by_platform(tags=get_supported(), sys_version=_sys_version())
        )
    else:
        filtered = _filter_by_platform(
            data, tags=get_supported(), sys_version=_sys_version()
        )

    flat = reduce(list.__add__, filtered.releases.values(), [])
    return sorted(flat, key=lambda r: (not r.is_sdist, r.version), reverse=True)


@cache
def _is_version_satisfied(specifier: str, sys_version) -> bool:
    """
    SpecifierSet("") matches anything, no need to artificially
    lock down versions at this point

    @see https://warehouse.readthedocs.io/api-reference/json.html
    @see https://packaging.pypa.io/en/latest/specifiers.html
    """
    specifiers = SpecifierSet(specifier or "")
    is_match = sys_version in specifiers

    log.debug("is_version_satisfied(info=i)%s in %s", sys_version, specifiers)
    return is_match


@D.pipes
def _is_platform_compatible(
    info: PyPI_Release, platform_tags: frozenset[PlatformTag], include_sdist=False
) -> bool:

    if "py2" in info.justuse.python_tag and "py3" not in info.justuse.python_tag:
        return False

    if not include_sdist and (
        ".tar" in info.justuse.ext or info.justuse.python_tag in ("cpsource", "sdist")
    ):
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

    given_platform_tags = (
        info.justuse.platform_tag.split(".") << map(PlatformTag) >> frozenset
    )

    if info.is_sdist and info.requires_python is not None:
        given_python_tag = {
            our_python_tag
            for p in info.requires_python.split(",")
            if Version(platform.python_version()) in SpecifierSet(p)
        }
    else:
        given_python_tag = set(info.justuse.python_tag.split("."))

    # print(supported_tags, given_python_tag)

    return any(supported_tags.intersection(given_python_tag)) and (
        (info.is_sdist and include_sdist)
        or any(given_platform_tags.intersection(platform_tags))
    )


def _is_compatible(
    info: PyPI_Release, sys_version, platform_tags, include_sdist=None
) -> bool:
    """Return true if the artifact described by 'info'
    is compatible with the current or specified system."""
    specifier = info.requires_python

    return (
        (not specifier or _is_version_satisfied(specifier, sys_version))
        and _is_platform_compatible(info, platform_tags, include_sdist)
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
            if "VERBOSE" in os.environ:
                log.debug(f"Applying aspect to {thing}.{name}")
            thing.__dict__[name] = decorator(obj)
    return thing


def _get_version(
    name: Optional[str] = None, package_name=None, /, mod=None
) -> Optional[Version]:
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
    if default is not mode.fastfail:
        return default  # TODO: write test for default
    else:
        raise exception
