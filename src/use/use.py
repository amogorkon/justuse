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

:author: Anselm Kiefner (amogorkon)
:author: David Reilly
:license: MIT
"""

# Structure of this module:
# 1) imports
# 2) setup of config, logging
# 3) global functions
# 4) ProxyModule and Use
# 5) initialization

# Read in this order:
# 1) initialization (instance of Use() is set as module on import)
# 2) use() dispatches to one of three __call__ methods, depending on first argument
# 3) from there, various global functions are called
# 4) a ProxyModule is always returned, wrapping the module that was imported


from __future__ import annotations

import ast
import asyncio
import atexit
import codecs
import hashlib
import importlib.util
import inspect
import linecache
import os
import re
import signal
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import zipfile
import zipimport
from collections import namedtuple
from enum import Enum
from functools import lru_cache as cache
from functools import singledispatch, update_wrapper
from importlib import metadata
from importlib.machinery import SourceFileLoader
from inspect import getsource, isclass, stack
from itertools import takewhile
from logging import DEBUG, INFO, NOTSET, WARN, StreamHandler, getLogger, root
from pathlib import Path
from subprocess import PIPE, run
from textwrap import dedent
from types import ModuleType
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union
from warnings import warn

import packaging
import requests
import toml
from furl import furl as URL
from packaging import tags
from packaging.specifiers import SpecifierSet
from packaging.version import Version as PkgVersion

# injected via initial_globals for testing, you can safely ignore this
test_config: str = locals().get("test_config", {})
test_version: str = locals().get("test_version", None)
__version__ = test_version or "0.4.2"

_reloaders: Dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

ModInUse = namedtuple("ModInUse", "name mod path spec frame")

# sometimes all you need is a sledge hammer..
def signal_handler(sig, frame) -> None:
    for reloader in _reloaders.values():
        reloader.stop()
    sig, frame
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves.
# # This is necessary to compare sys.version_info with Version and make some tests more elegant, amongst other things.
class Version(PkgVersion):
    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if not (versionstr or major or minor or patch):
            raise ValueError(
                "Version must be initialized with either a string or major, minor and patch"
            )
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        return super().__init__(versionstr)

    def __iter__(self):
        yield from self.release


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


class PlatformTag(namedtuple("PlatformTag", ["platform"])):
    def __str__(self):
        return self.platform

    def __repr__(self):
        return self.platform


# singledispatch for methods
def methdispatch(func) -> Callable:
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


# Really looking forward to actual builtin sentinel values..
mode = Enum("Mode", "fastfail")

# defaults
config = {"version_warning": True, "debugging": False, "use_db": False}

# initialize logging
root.addHandler(StreamHandler(sys.stderr))
root.setLevel(NOTSET)
if "DEBUG" in os.environ or "pytest" in getattr(
    sys.modules.get("__init__", ""), "__file__", ""
):
    root.setLevel(DEBUG)
    test_config["debugging"] = True
else:
    root.setLevel(INFO)

# TODO: log to file
log = getLogger(__name__)


class Hash(Enum):
    sha256 = hashlib.sha256


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


# This is a collection of the messages directed to the user.
# How it works is quite magical - the lambdas prevent the f-strings from being prematuraly evaluated, and are only evaluated once returned.
# Fun fact: f-strings are firmly rooted in the AST.
class Message(Enum):
    not_reloadable = (
        lambda name: f"Beware {name} also contains non-function objects, it may not be safe to reload!"
    )
    couldnt_connect_to_db = (
        lambda e: f"Could not connect to the registry database, please make sure it is accessible. ({e})"
    )
    version_warning = (
        lambda max_version: f"""Justuse is version {Version(__version__)}, but there is a newer version {max_version} available on PyPI.
To find out more about the changes check out https://github.com/amogorkon/justuse/wiki/What's-new
Please consider upgrading via 'python -m pip install -U justuse'"""
    )
    cant_use = (
        lambda thing: f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}."
    )
    web_error = (
        lambda url, response: f"Could not load {url} from the interwebs, got a {response.status_code} error."
    )
    no_validation = (
        lambda url, hash_algo, this_hash: f"""Attempting to import from the interwebs with no validation whatsoever!
To safely reproduce: use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')"""
    )
    aspectize_builtins_warning = (
        lambda: "Applying aspects to builtins may lead to unexpected behaviour, but there you go.."
    )
    version_warning = (
        lambda name, target_version, this_version: f"{name} expected to be version {target_version}, but got {this_version} instead"
    )
    ambiguous_name_warning = (
        lambda name: f"Attempting to load the package '{name}', if you rather want to use the local module: use(use.Path('{name}.py'))"
    )
    no_version_provided = (
        lambda: "No version was provided, even though auto_install was specified! Trying to load classically installed package instead."
    )
    classically_imported = (
        lambda name, this_version: f'Classically imported \'{name}\'. To pin this version: use("{name}", version="{this_version}")'
    )
    pebkac_missing_hash = (
        lambda package_name, version, hashes: f"""Failed to auto-install '{package_name}' because hash_value is missing. This may work:
use("{package_name}", version="{version}", hashes={hashes!r}, modes=use.auto_install)"""
    )
    pebkac_unsupported = (
        lambda package_name: f"We could not find any version or release for {package_name} that could satisfy our requirements!"
    )
    pip_json_mess = (
        lambda package_name, target_version: f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI."
    )
    no_version_or_hash_provided = (
        lambda name, package_name, version, hashes: f"""Please specify version and hash for auto-installation of '{package_name}'.
To get some valuable insight on the health of this package, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version: use("{name}", version="{version}", hashes={hashes!r}, modes=use.auto_install)
"""
    )


# Since we have quite a bit of functional code that black would turn into a sort of arrow antipattern with lots of ((())),
# we use @pipes to basically enable polish notation which allows us to avoid most parentheses.
# source >> func(args) is equivalent to func(source, args) and
# source << func(args) is equivalent to func(args, source), which can be chained arbitrarily.
# Rules:
# 1) apply pipes only to 3 or more nested function calls
# 2) no pipes on single lines, since mixing << and >> is just confusing (also, having pipes on different lines has other benefits beside better readability)
# 3) don't mix pipes with regular parenthesized function calls, that's just confusing
# See https://github.com/robinhilliard/pipes/blob/master/pipeop/__init__.py for details and credit.
class _PipeTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if not isinstance(node.op, (ast.LShift, ast.RShift)):
            return node
        if not isinstance(node.right, ast.Call):
            return self.visit(
                ast.Call(
                    func=node.right,
                    args=[node.left],
                    keywords=[],
                    starargs=None,
                    kwargs=None,
                    lineno=node.right.lineno,
                    col_offset=node.right.col_offset,
                )
            )
        node.right.args.insert(
            0 if isinstance(node.op, ast.RShift) else len(node.right.args), node.left
        )
        return self.visit(node.right)


def pipes(func_or_class):
    if isclass(func_or_class):
        decorator_frame = stack()[1]
        ctx = decorator_frame[0].f_locals
        first_line_number = decorator_frame[2]
    else:
        ctx = func_or_class.__globals__
        first_line_number = func_or_class.__code__.co_firstlineno
    source = getsource(func_or_class)
    tree = ast.parse(dedent(source))
    ast.increment_lineno(tree, first_line_number - 1)
    source_indent = sum(1 for _ in takewhile(str.isspace, source)) + 1

    for node in ast.walk(tree):
        if hasattr(node, "col_offset"):
            node.col_offset += source_indent
    tree.body[0].decorator_list = [
        d
        for d in tree.body[0].decorator_list
        if isinstance(d, ast.Call)
        and d.func.id != "pipes"
        or isinstance(d, ast.Name)
        and d.id != "pipes"
    ]

    tree = _PipeTransformer().visit(tree)
    code = compile(
        tree, filename=(ctx["__file__"] if "__file__" in ctx else "repl"), mode="exec"
    )
    exec(code, ctx)
    return ctx[tree.body[0].name]


@cache
def get_supported() -> FrozenSet[PlatformTag]:
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, pkg_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    items: List[PlatformTag] = []
    from pip._internal.utils import compatibility_tags  # type: ignore

    for tag in compatibility_tags.get_supported():
        items.append(PlatformTag(platform=tag.platform))
    for tag in packaging.tags._platform_tags():
        items.append(PlatformTag(platform=str(tag)))

    tags = frozenset(items + ["any"])
    log.error(str(tags))
    return tags


def archive_meta(artifact_path):
    with zipfile.ZipFile(artifact_path) as archive:
        names = sorted(
            file.filename for file in archive.filelist if not file.filename.endswith("pyi")
        )
        meta = dict(
            map(
                lambda n: (
                    Path(n).stem,
                    (m := archive.open(n), str(m.read(), "UTF-8").splitlines(), m.close())[
                        1
                    ],
                ),
                (
                    n
                    for n in names
                    if re.search("(dist-info|-INFO|\\.txt$|(^|/)[A-Z0-9_-]+)$", n)
                ),
            )
        )
        name = next(
            l.partition(": ")[-1]
            for l in meta["METADATA" if "METADATA" in meta else "PKG-INFO"]
            if l.startswith("Name: ")
        )
        md_lines = next(i for i in meta.values() if "Metadata-Version" in str(i))
        info = {
            p[0].lower().replace("-", "_"): p[2]
            for p in (l.partition(": ") for l in md_lines)
        }
        meta.update(info)
        meta["names"] = names
        if "top_level" not in meta:
            relpath = sorted(
                [*(n for n in names if re.search("[^/]+([.][^/]+|[-][^/]+)$", n))],
                key=lambda n: ("__init__.py" not in n, len(n)),
            )[0]

            rel2 = relpath.replace("-", "!").split("!")[0]
            rel3 = Path(rel2)
            rel4 = rel3.parts
            if rel3.stem == "__init__":
                import_name = rel4[0:-1]
            else:
                import_name = rel4[0:-1] + [rel3.stem]
            meta["top_level"] = [import_name]
            meta["import_name"] = import_name
        else:
            top_level, name, = (
                meta["top_level"][0],
                meta["name"],
            )
            import_name = (name,) if (top_level == name) else (top_level, name)
            meta["import_name"] = import_name
            relpath = sorted(
                [*(n for n in names if re.search("[^/]+([.][^/]+|[-][^/]+)$", n))],
                key=lambda n: (
                    not n.startswith(import_name),
                    not n.endswith("__init__.py"),
                    len(n),
                ),
            )[0]

        meta["import_relpath"] = relpath
        return meta


def _clean_sys_modules(package_name) -> None:
    del_mods = dict(
        [
            (k, v.__spec__.loader)
            for k, v in sys.modules.items()
            if getattr(v, "__spec__", None)
            and isinstance(v.__spec__.loader, SourceFileLoader)
            and (k.startswith(f"{package_name}.") or k == package_name)
        ]
    )
    for k in del_mods:
        del sys.modules[k]


def _venv_root(package_name, version) -> Path:
    venv_root = (
        Path.home() / ".justuse-python" / "venv" / package_name / (version or "0.0.0")
    )
    if not venv_root.exists():
        venv_root.mkdir(parents=True)
    return venv_root


def _venv_is_win() -> bool:
    return sys.platform.lower().startswith("win")


def _no_pebkac(name, package_name, target_version, hash_algo, hashes) -> bool:
    if target_version and not hashes:  # let's try to be helpful
        data = _get_filtered_data(_get_package_data(package_name))
        version = target_version
        entry = data["releases"][str(target_version)][-1]
        that_hash = entry["digests"][hash_algo.name]
        hit = (version, that_hash)
        log.info(f"{hit=} from Use._find_matching_artifact")
        if that_hash:
            if that_hash is not None:
                hashes.add(that_hash)
            raise RuntimeWarning(Message.pebkac_missing_hash(package_name, version, hashes))
        raise RuntimeWarning(
            f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
        )
    elif not target_version and hashes:
        raise RuntimeWarning(
            f"Failed to auto-install '{package_name}' because no version was specified."
        )
    elif not target_version:
        # let's try to make an educated guess and give a useful suggestion
        data = _get_filtered_data(_get_package_data(package_name))
        for ver, infos in reversed(data["releases"].items()):
            entry = infos[-1]
            if not hashes or entry["digests"][hash_algo.name] in hashes:

                hash_value = entry["digests"][hash_algo.name]
                version = ver
                hit = (version, hash_value)

        if not hash_value:
            raise RuntimeWarning(Message.pebkac_unsupported(package_name))

        if not target_version:
            hashes.add(hash_value)
            raise RuntimeWarning(
                Message.no_version_or_hash_provided(name, package_name, version, hashes)
            )
    return True


def _update_hashes(
    package_name,
    target_version,
    version,
    default,
    hash_algo,
    hashes,
    all_hashes,
    home,
) -> None:
    found = None
    try:
        data = _get_filtered_data(_get_package_data(package_name))
        infos = data["releases"][str(target_version)]
        for entry in infos:
            url = URL(entry["url"])
            path = home / "packages" / Path(url.asdict()["path"]["segments"][-1]).name
            log.error("url = %s", url)
            entry["version"] = str(target_version)
            log.debug(f"looking at {entry=}")
            assert isinstance(all_hashes, set)
            all_hashes.add(that_hash := entry["digests"].get(hash_algo.name))
            if hashes.intersection(all_hashes):

                found = (entry, that_hash)
                hit = VerHash(version, that_hash)
                log.info(f"Matches user hash: {entry=} {hit=}")
                break
        if found is None:
            return _fail_or_default(
                default,
                AutoInstallationError,
                f"Tried to auto-install ({package_name=!r}) with {target_version=!r} but failed because none of the available hashes ({all_hashes=!r}) match the expected hash ({hashes=!r}).",
            )
        entry, that_hash = found
        if that_hash is not None:
            assert isinstance(hashes, set)
            hashes.add(that_hash)
    except KeyError as be:  # json issuesa
        msg = f"request to https://pypi.org/pypi/{package_name}/{target_version}/json lead to an error: {be}"
        raise RuntimeError(msg) from be
    exc = None
    if exc:
        return _fail_or_default(
            default,
            AutoInstallationError,
            Message.pip_json_mess(package_name, target_version),
        )


def isfunction(x: Any) -> bool:
    return inspect.isfunction(x)


def ismethod(x: Any) -> bool:
    return inspect.isfunction(x)


def ismethod(x: Any) -> bool:
    return inspect.isfunction(x)


# decorators for callable classes require a completely different approach i'm afraid.. removing the check should discourage users from trying
# def isclass(x):
#     return inspect.isclass(x) and hasattr(x, "__call__")
def _parse_filename(filename) -> dict:
    """Match the filename and return a dict of parts.
    >>> parse_filename("numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl")
    {'distribution': 'numpy', 'version': '1.19.5', 'build_tag', 'python_tag': 'cp36', 'abi_tag': 'cp36m', 'platform_tag': 'macosx_10_9_x86_64', 'ext': 'whl'}
    """
    # Filename as API, seriously WTF...
    assert isinstance(filename, str)
    match = re.match(
        "(?P<distribution>.*)-"
        "(?P<version>.*)"
        "(?:-(?P<build_tag>.*))?-"
        "(?P<python_tag>.*)-"
        "(?P<abi_tag>.*)-"
        "(?P<platform_tag>.*)\\."
        "(?P<ext>whl|zip|egg|tar|tar\\.gz)",
        filename,
    )
    return match.groupdict() if match else {}


def _load_venv_mod(
    name_prefix, name, version=None, artifact_path=None, url=None, out_info=None
) -> ModuleType:
    venv_root = _venv_root(name_prefix or name, version or "0.0.0")
    venv_bin = venv_root / "bin"
    python_exe = Path(sys.executable).name
    current_path = os.environ.get("PATH")
    venv_path_var = f"{venv_bin}{os.path.pathsep}{current_path}"
    run([python_exe, "-m", "venv", venv_root])
    install_item = name_prefix or name
    package_name = name_prefix or name
    if not (artifact_path and artifact_path.exists()):
        install_item = install_item or package_name or name
        if version:
            install_item += f"=={version}"
    if isinstance(url, str):
        url = URL(url)
    filename = artifact_path.name if artifact_path else None
    if url:
        filename = url.asdict()["path"]["segments"][-1]
    if filename and not artifact_path:
        artifact_path = sys.modules["use"].home / "packages" / filename
    log.info("Installing %s using pip", install_item)
    for p in venv_root.rglob("**/bin/python"):
        venv_bin = Path(p).parent
        python_exe = p.name
    for p in venv_root.rglob("**/python.exe"):
        venv_bin = Path(p).parent
        python_exe = p.name

    pip_args = [
        str(venv_bin / python_exe),
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-color",
        "install",
        "--pre",
        "-v",
        "-v",
        "--prefer-binary",
        "--exists-action",
        "b",
        "--no-build-isolation",
        "--no-cache-dir",
        "--no-compile",
        "--no-warn-script-location",
        "--no-warn-conflicts",
    ]
    for extra_args in ([], ["--force-reinstall"]):
        output = run(
            (
                ["cmd.exe", "/C", "set", f"PATH={venv_path_var}", "&"]
                if _venv_is_win()
                else ["env", f"PATH={venv_path_var}"]
            )
            + pip_args
            + extra_args
            + [install_item],
            encoding="UTF-8",
            stdout=PIPE,
            stderr=PIPE,
        )
        output.check_returncode()
        match = re.compile(
            "Added (?P<package_name>[^= ]+)(?:==(?P<version>[^ ]+)|) from (?P<url>[^# ,\n]+(?:/(?P<filename>[^#, \n/]+)))(?:#(\\S*)|)(?=\\s)",
            re.DOTALL,
        ).search(output.stdout)
        if match or (artifact_path and artifact_path.exists()):
            break
    assert match or (artifact_path and artifact_path.exists())
    info = match.groupdict() if match else {}
    filename = filename or info["filename"] or Path(artifact_path).name
    if info:
        url = url or URL(info["url"])
        filename = filename or info["filename"]
        package_name = package_name or info["package_name"]
        version = info["version"] or version
        if out_info:
            out_info.update(info)
    if filename and not artifact_path:
        artifact_path = sys.modules["use"].home / "packages" / filename
    if not artifact_path.exists():
        artifact_path.write_bytes(requests.get(url).content)
        assert artifact_path.exists()
    orig_cwd = Path.cwd()
    meta = archive_meta(artifact_path)
    meta.update(info)
    relp = meta["import_relpath"]
    module_path = [*venv_root.rglob(f"**/{relp}")][0]
    name_segments = ".".join(relp.split(".")[0:-1]).split("-")[0].replace("/", ".")
    name_prefix, _, name = name_segments.rpartition(".")
    installation_path = module_path
    for _ in range(len(name_segments.split("."))):
        installation_path = installation_path.parent
    try:
        log.error("installation_path = %s", installation_path)
        os.chdir(str(installation_path))
        with open(module_path, "rb") as f:
            if out_info is not None:
                out_info.update(
                    {
                        "artifact_path": artifact_path,
                        "installation_path": installation_path,
                        "module_path": module_path,
                        "package": package_name,
                        "package_name": package_name,
                        "name_prefix": name_prefix,
                        "name": name,
                        "url": url,
                        "version": version,
                        "info": info,
                        **meta,
                    }
                )
            return _load_venv_entry(
                package_name=name_prefix, name=name, module_path=module_path
            )
    finally:
        os.chdir(orig_cwd)


def _load_venv_entry(package_name, name, module_path) -> ModuleType:
    mod = None
    try:
        _clean_sys_modules(name)
        _clean_sys_modules(package_name)
        log.info(
            "load_venv_entry package_name=%s name=%s module_path=%s",
            package_name,
            name,
            module_path,
        )
        with open(module_path, "rb") as code_file:
            return (
                mod := _build_mod(
                    name=name,
                    code=code_file.read(),
                    module_path=Path(module_path),
                    initial_globals={},
                    aspectize={},
                    package_name=package_name,
                )
            )
    finally:
        log.info("load_venv_entry returning mod=%s", mod)
        _clean_sys_modules(name)
        _clean_sys_modules(package_name)


@cache
def _get_package_data(package_name) -> Dict[str, str]:
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    if response.status_code == 404:
        raise ImportError(
            "Package {package_name!r} is not available on pypi; are you sure it exists?"
        )
    elif response.status_code != 200:
        raise RuntimeWarning(
            f"Something bad happened while contacting PyPI for info on {package_name} ( {response.status_code} ), which we tried to look up because a matching hashes for the auto-installation was missing."
        )
    data: dict = response.json()
    for v, infos in data["releases"].items():
        for info in infos:
            info["version"] = v
            info.update(_parse_filename(info["filename"]))
    return data


def _get_filtered_data(data) -> Dict[str, str]:
    filtered = {"urls": [], "releases": {}}
    for ver, infos in data["releases"].items():
        filtered["releases"][ver] = []
        for info in infos:
            info["version"] = ver
            if not _is_compatible(info, hash_algo=Hash.sha256.name, include_sdist=False):
                continue
            filtered["urls"].append(info)
            filtered["releases"][ver].append(info)
    for ver in data["releases"].keys():
        if not filtered["releases"][ver]:
            del filtered["releases"][ver]
    return filtered


def _is_version_satisfied(info, sys_version) -> bool:
    """
    SpecifierSet("") matches anything, no need to artificially lock down versions at this point

    @see https://warehouse.readthedocs.io/api-reference/json.html
    @see https://packaging.pypa.io/en/latest/specifiers.html
    """
    specifier = info.get("requires_python", "")
    # f.. you PyPI
    return sys_version in SpecifierSet(specifier or "")


def _is_platform_compatible(
    info,
    platform_tags,
    include_sdist=False,
) -> bool:
    assert isinstance(info, dict)
    assert isinstance(platform_tags, frozenset)
    if "platform_tag" not in info:
        info.update(_parse_filename(info["filename"]))
    our_python_tag = tags.interpreter_name() + tags.interpreter_version()
    python_tag = info.get("python_tag", "")
    if python_tag == "py3":
        python_tag = our_python_tag
    platform_tag = info.get("platform_tag", "")
    platform_srs = {*map(str, platform_tags)}
    is_sdist = (
        info["packagetype"] == "sdist"
        or info["python_version"] == "source"
        or info.get("abi_tag", "") == "none"
    )
    reject = "py2" in info["filename"] or ".tar" in info["filename"]
    if not platform_tag:
        platform_tag = "any"
    if not python_tag:
        python_tag = "cp" + info["python_version"].replace(".", "")
    for one_platform_tag in platform_tag.split("."):
        matches_platform = one_platform_tag in platform_srs
        matches_python = our_python_tag == python_tag
        if "VERBOSE" in os.environ:
            log.debug(
                f"({matches_platform=} from {one_platform_tag=} and {matches_python=} from {python_tag=}) or ({include_sdist=} and {is_sdist=}) and not {reject=}: {info['filename']}"
            )
        if (
            (matches_platform and matches_python)
            or (include_sdist and is_sdist)
            and not reject
        ):
            return True
    return False


def _is_compatible(
    info,
    hash_algo=Hash.sha256.name,
    sys_version=None,
    platform_tags=frozenset(),
    include_sdist=False,
) -> bool:
    """Return true if the artifact described by 'info'
    is compatible with the current or specified system."""
    assert isinstance(info, dict)
    if not sys_version:
        sys_version = Version(".".join(map(str, sys.version_info[0:3])))
    if not platform_tags:
        platform_tags = get_supported()
    if "platform_tag" not in info:
        info.update(_parse_filename(info["filename"]))
    assert isinstance(sys_version, Version)
    return (
        _is_version_satisfied(info, sys_version)
        and _is_platform_compatible(info, platform_tags, include_sdist)
        and not info["yanked"]
        and not info["filename"].endswith(".tar")
        and not info["filename"].endswith(".tar.gz")
    )


def dict_factory(cursor, row) -> Dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


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
            log.debug(f"Applying aspect to {thing}.{name}")
            thing.__dict__[name] = decorator(obj)
    return thing


def _get_version(name=None, package_name=None, /, mod=None) -> Optional[Version]:
    assert name is None or isinstance(name, str)
    version: Optional[Union[Callable[...], Version] | Version | str] = None
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


def _build_mod(
    *,
    name,
    code,
    initial_globals: Optional[Dict[str, Any]],
    module_path,
    aspectize,
    aspectize_dunders=False,
    package_name=None,
) -> ModuleType:
    name_qual = (
        package_name + "." + name
        if (package_name and name and package_name != name)
        else name
    ).replace(".__init__", "")
    from pathlib import Path

    module_path = (
        Path("%s" % module_path) if not isinstance(module_path, Path) else module_path
    )

    if "__init__" in module_path.stem:
        mod = ModuleType(name_qual + ".__init__")
    else:
        mod = ModuleType(name_qual)

    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__package__ = package_name
    mod.__name__ = name

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
    try:
        exec(compile(code, f"<{name}>", "exec"), vars(mod))
    except:
        try:
            exec(compile(code, module_path, "exec"), vars(mod))
        except:  # reraise anything without handling - clean and simple.
            raise
    for (check, pattern), decorator in aspectize.items():
        _apply_aspect(mod, check, pattern, decorator, aspectize_dunders=aspectize_dunders)
    return mod


def _ensure_proxy(mod) -> ProxyModule:
    if mod.__class__ is not ModuleType:
        return mod
    return ProxyModule(mod)


def _fail_or_default(default, exception, msg):
    if default is not mode.fastfail:
        return default  # TODO: write test for default
    else:
        raise exception(msg)


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
            current_filehash = hashlib.blake2b(code).hexdigest()
            if current_filehash != last_filehash:
                try:
                    mod = _build_mod(
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
                current_filehash = hashlib.blake2b(code).hexdigest()
                if current_filehash != last_filehash:
                    try:
                        mod = _build_mod(
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


class Use(ModuleType):
    # MODES to reduce signature complexity
    # enum.Flag wasn't viable, but this alternative is actually pretty cool
    auto_install = 2 ** 0
    fatal_exceptions = 2 ** 1
    reloading = 2 ** 2
    aspectize_dunders = 2 ** 3

    def __init__(self):
        # TODO for some reason removing self._using isn't as straight forward..
        self._using = _using
        self.home: Path
        self._hacks = {}

        self._set_up_files_and_directories()
        # might run into issues during testing otherwise
        if not test_version:
            try:
                self.registry = sqlite3.connect(self.home / "registry.db").cursor()
                self.registry.execute("PRAGMA foreign_keys=ON")
                self.registry.execute("PRAGMA auto_vacuum = FULL")
            except Exception as e:
                raise RuntimeError(Message.couldnt_connect_to_db(e))
        else:
            self.registry = sqlite3.connect(":memory:").cursor()
        self.registry.row_factory = dict_factory

        self._set_up_registry()
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
                name = "justuse"
                response = requests.get(f"https://pypi.org/pypi/{name}/json")
                data = response.json()
                max_version = max(Version(version) for version in data["releases"].keys())
                target_version = max_version
                this_version = __version__
                if Version(this_version) < target_version:
                    warn(
                        Message.version_warning(name, target_version, this_version),
                        VersionWarning,
                    )
            except:
                if test_version:
                    raise
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
        """Update the registry to contain the package's metadata."""
        # version = str(version) if version else "0.0.0"
        # assert version not in ("None", "null", "")
        assert isinstance(version, Version)

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
        self._using[name] = ModInUse(name, mod, path, spec, frame)

    @methdispatch
    def __call__(self, thing, /, *args, **kwargs):
        raise NotImplementedError(Message.cant_use(thing))

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
        modes=0,
    ) -> ModuleType:
        log.debug(f"use-url: {url}")
        exc = None
        assert url is not None, f"called with url == {url!r}"
        assert url != "None", f"called with url == {url!r}"

        assert hash_algo in Hash, f"{hash_algo} is not a valid hashing algorithm!"

        aspectize = aspectize or {}
        response = requests.get(str(url))
        if response.status_code != 200:
            raise ImportError(Message.web_error())
        this_hash = hash_algo.value(response.content).hexdigest()
        if hash_value:
            if this_hash != hash_value:
                return _fail_or_default(
                    default,
                    Use.UnexpectedHash,
                    f"{this_hash} does not match the expected hash {hash_value} - aborting!",
                )
        else:
            warn(Message.no_validation(url, hash_algo, this_hash), NoValidationWarning)
        name = str(url)

        try:
            mod = _build_mod(
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
            return _fail_or_default(default, ImportError, exc)

        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        if as_import:
            assert isinstance(
                as_import, str
            ), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            assert as_import.isidentifier(), "as_import must be a valid identifier."
            sys.modules[as_import] = mod
        return _ensure_proxy(mod)

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
        modes=0,
    ) -> Optional[ModuleType]:
        """Import a module from a path.

        https://github.com/amogorkon/justuse/wiki/Use-Path

        Args:
            path ([type]): must be a pathlib.Path
            initial_globals ([type], optional): Dict that should be globally available to the module before executing it. Defaults to None.
            default ([type], optional): Return instead if an exception is encountered.
            aspectize ([type], optional): Aspectize callables. Defaults to None.
            modes (int, optional): [description]. Defaults to 0; Acceptable mode for this variant: use.reloading.

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
            return _fail_or_default(default, ImportError, f"Can't import directory {path}")

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
                return _fail_or_default(
                    default,
                    NotImplementedError,
                    "Can't determine a relative path from a virtual file.",
                )
            path = source_dir.joinpath(path).resolve()
            if not path.exists():
                return _fail_or_default(default, ImportError, f"Sure '{path}' exists?")
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
                        module_path=str(path.resolve()),
                        aspectize=aspectize,
                    )
                except:
                    exc = traceback.format_exc()
                if exc:
                    return _fail_or_default(default, ImportError, exc)
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
                    warn(Message.not_reloadable(), NotReloadableWarning)
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
                        module_path=str(path),
                        aspectize=aspectize,
                    )
                except:
                    del self._using[name]
                    exc = traceback.format_exc()
        except:
            exc = traceback.format_exc()
        finally:
            # let's not confuse the user and restore the cwd to the original in any case
            os.chdir(original_cwd)
        if exc:
            return _fail_or_default(default, ImportError, exc)
        if as_import:
            assert isinstance(
                as_import, str
            ), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            sys.modules[as_import] = mod
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return _ensure_proxy(mod)

    def _import_builtin(self, name, spec, default, aspectize):
        try:
            mod = spec.loader.create_module(spec)
            spec.loader.exec_module(mod)  # ! => cache
            if aspectize:
                warn(Message.aspectize_builtins_warning(), RuntimeWarning)
            for (check, pattern), decorator in aspectize.items():
                _apply_aspect(
                    mod, check, pattern, decorator, aspectize_dunders=self.aspectize_dunders
                )
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=mod, spec=spec, frame=frame)
            return _ensure_proxy(mod)
        except:
            exc = traceback.format_exc()
        return _fail_or_default(default, ImportError, exc)

    def _import_classical_install(
        self,
        *,
        name,
        module_name,
        spec,
        target_version,
        default,
        aspectize,
        fatal_exceptions,
        package_name=None,
    ):
        fatal_exceptions |= "ERRORS" in os.environ
        exc = None
        try:
            # TODO - use spec?
            mod = importlib.import_module(module_name)  # ! => cache
            for (check, pattern), decorator in aspectize.items():
                _apply_aspect(
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
                    Message.classically_imported(name, metadata.version(name)),
                    AmbiguityWarning,
                )
                return _ensure_proxy(mod)
        except:
            if fatal_exceptions:
                raise
            exc = traceback.format_exc()
        if exc:
            return _fail_or_default(default, ImportError, exc)
        # we only enforce versions with auto-install
        this_version = _get_version(name, package_name, mod=mod)
        if not this_version:
            log.warning(f"Cannot find version for {name=}, {mod=}")
        elif not target_version:
            warn("No version was specified", AmbiguityWarning)
        elif target_version != this_version:
            warn(Message.version_warning(name, target_version, this_version), VersionWarning)
        for (check, pattern), decorator in aspectize.items():
            _apply_aspect(
                mod,
                check,
                pattern,
                decorator,
                aspectize_dunders=bool(Use.aspectize_dunders & self.modes),
            )
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return _ensure_proxy(mod)

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

        Raises:
            RuntimeWarning: May be raised if the auto-installation of the package fails for some reason.

        Returns:
            Optional[ModuleType]: Module if successful, default as specified otherwise.
        """
        log.debug(f"use-str: {name}")
        self.modes = modes
        aspectize = aspectize or {}

        if isinstance(hashes, str):
            hashes = set([hashes])
        hashes = set(hashes) if hashes else set()

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

        mod = None

        # The "try and guess" behaviour is due to how classical imports work,
        # which is inherently ambiguous, but can't really be avoided for packages.
        # let's first see if the user might mean something else entirely
        if any(Path(".").glob(f"{name}.py")):
            warn(Message.ambiguous_name_warning(name), AmbiguityWarning)
        spec = None
        that_hash = None
        all_hashes = set()

        if name in self._using:
            spec = self._using[name].spec
        elif not auto_install:
            spec = importlib.util.find_spec(name)

        {
            (False, False, False, False): lambda **kwargs: None,
            (False, False, False, True): lambda **kwargs: None,
            (False, False, True, False): lambda **kwargs: None,
            (False, True, False, False): lambda **kwargs: None,
            (True, False, False, False): lambda **kwargs: None,
            (False, False, True, True): lambda **kwargs: None,
            (False, True, True, False): lambda **kwargs: None,
            (True, True, False, False): lambda **kwargs: None,
            (True, False, False, True): lambda **kwargs: None,
            (True, False, True, False): lambda **kwargs: None,
            (False, True, False, True): lambda **kwargs: None,
            (False, True, True, True): lambda **kwargs: None,
            (True, False, True, True): lambda **kwargs: None,
            (True, True, False, True): lambda **kwargs: None,
            (True, True, True, False): lambda **kwargs: None,
            (True, True, True, True): lambda **kwargs: None,
        }[(bool(version), bool(hashes), bool(spec), bool(auto_install))](
            name=name, version=version, hashes=hashes, spec=spec
        )

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
                    name=name,
                    module_name=module_name,
                    spec=spec,
                    target_version=target_version,
                    default=default,
                    aspectize=aspectize,
                    fatal_exceptions=fatal_exceptions,
                )

            this_version = _get_version(name, package_name)
            log.error("this_version=%s, target_version=%s", this_version, target_version)

            if this_version == target_version:
                assert False, "we need a test for this or remove the code block"
                if not (version):
                    warn(Message.no_version_provided(), AmbiguityWarning)
                mod = self._import_classical_install(
                    name=name,
                    module_name=module_name,
                    spec=spec,
                    target_version=target_version,
                    default=default,
                    aspectize=aspectize,
                    fatal_exceptions=fatal_exceptions,
                    package_name=package_name,
                )
                warn(
                    Message.classically_imported(),
                    AmbiguityWarning,
                )
                return _ensure_proxy(mod)
            # wrong version => wrong spec
            this_version = _get_version(mod=mod)
            if this_version != target_version:
                spec = None
                log.warning(
                    f"Setting {spec=}, since " f"{target_version=} != {this_version=}"
                )
        else:
            if not auto_install:
                return _fail_or_default(
                    default,
                    ImportError,
                    f"Could not find any installed package '{name}' and auto_install was not requested.",
                )

            if _no_pebkac(name, package_name, target_version, hash_algo, hashes):
                pass

            # if it's a pure python package, there is only an artifact, no installation
            query = self.registry.execute(
                "SELECT id, installation_path FROM distributions WHERE name=? AND version=?",
                (name, version),
            ).fetchone()

            # TODO: looks like a good place for a JOIN
            if query:
                installation_path = query["installation_path"]
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
                _update_hashes(
                    package_name,
                    target_version,
                    version,
                    default,
                    hash_algo,
                    hashes,
                    all_hashes,
                    self.home,
                )

        if path:
            # trying to import directly from zip
            try:
                importer = zipimport.zipimporter(path)
                mod = importer.load_module(module_name)
                print("Direct zipimport of", name, "successful.")
            except:
                if config["debugging"]:
                    log.debug(traceback.format_exc())
                print("Direct zipimport failed, attempting to extract and load manually...")

        installation_path = (path.parent / path.stem) if path else None

        if not mod:
            out_info = {}
            mod = _load_venv_mod(
                name_prefix=package_name,
                name=name,
                version=version,
                artifact_path=path,
                url=url,
                out_info=out_info,
            )
            path = out_info["artifact_path"]
            installation_path = out_info["installation_path"]
            package_name = out_info["package_name"]
            name = out_info["name"]
            url = out_info["url"]
            this_version = Version(out_info["version"])
        self._save_module_info(
            name=name,
            version=this_version,
            artifact_path=path,
            hash_value=that_hash,
            installation_path=installation_path,
        )
        for (check, pattern), decorator in aspectize.items():
            if mod is not None:
                _apply_aspect(
                    mod,
                    check,
                    pattern,
                    decorator,
                    aspectize_dunders=bool(Use.aspectize_dunders & modes),
                )
        log.error("_use_str: mod=%s", mod)
        try:
            return mod
        finally:
            frame = inspect.getframeinfo(inspect.currentframe())
            if frame:
                self._set_mod(name=name, mod=mod, frame=frame)


use = Use()
use.__dict__.update(globals())
if not test_version:
    sys.modules["use"] = use
