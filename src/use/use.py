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
import codecs
import hashlib
import importlib.util
import inspect
import linecache
import os
import platform
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
from collections import namedtuple
from enum import Enum
from functools import lru_cache as cache
from functools import partial, partialmethod, reduce, singledispatch, update_wrapper
from importlib import metadata
from importlib.abc import Finder, Loader
from importlib.machinery import ModuleSpec, SourceFileLoader
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from importlib.util import find_spec
from inspect import isfunction, ismethod  # for aspectizing, DO NOT REMOVE
from itertools import chain, takewhile
from logging import DEBUG, INFO, NOTSET, WARN, StreamHandler, getLogger, root
from pathlib import Path, PureWindowsPath, WindowsPath
from pprint import pformat
from subprocess import PIPE, run
from textwrap import dedent
from types import FrameType, ModuleType
from typing import Any, Callable, Optional, Union
from warnings import warn

import furl
import packaging
import requests
import toml
from beartype import beartype
from furl import furl as URL
from icontract import ensure, invariant, require
from packaging import tags
from packaging.specifiers import SpecifierSet
from pip._internal.utils import compatibility_tags

cwd = Path("")
os.chdir(Path(__file__).parent)
import pypi_model

use = sys.modules.get(__name__)
from pypi_model import PyPI_Project, PyPI_Release, Version

os.chdir(cwd)

#% Constants and Initialization

# injected via initial_globals for testing, you can safely ignore this
test_config: str = locals().get("test_config", {})
test_version: str = locals().get("test_version", None)
__name__ = "use.use"
__package__ = "use"
__path__ = __file__
__spec__ = ModuleSpec("use.use", loader=SourceFileLoader(fullname="use", path=__file__))
__spec__.submodule_search_locations = [Path(__file__).parent]
__version__ = test_version or "0.5.0"

_reloaders: dict["ProxyModule", Any] = {}  # ProxyModule:Reloader
_aspects = {}
_using = {}

ModInUse = namedtuple("ModInUse", "name mod path spec frame")
NoneType = type(None)


# Really looking forward to actual builtin sentinel values..
mode = Enum("Mode", "fastfail")

# defaults
config = {"version_warning": True, "debugging": False, "use_db": True}

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
#%% Pipes

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
    if inspect.isclass(func_or_class):
        decorator_frame = inspect.stack()[1]
        ctx = decorator_frame[0].f_locals
        first_line_number = decorator_frame[2]
    else:
        ctx = func_or_class.__globals__
        first_line_number = func_or_class.__code__.co_firstlineno
    source = inspect.getsource(func_or_class)
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


#% Helper Functions

# keyword args from inside the called function!
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


# singledispatch for methods
def methdispatch(func) -> Callable:
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        # so we can dispatch on None
        if len(args) == 1:
            args = args + (None,)
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


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
    use_version_warning = (
        lambda max_version: f"""Justuse is version {Version(__version__)}, but there is a newer version {max_version} available on PyPI.
To find out more about the changes check out https://github.com/amogorkon/justuse/wiki/What's-new
Please consider upgrading via
python -m pip install -U justuse
"""
    )
    cant_use = (
        lambda thing: f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}."
    )
    web_error = (
        lambda url, response: f"Could not load {url} from the interwebs, got a {response.status_code} error."
    )
    no_validation = (
        lambda url, hash_algo, this_hash: f"""Attempting to import from the interwebs with no validation whatsoever!
To safely reproduce:
use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')"""
    )
    version_warning = (
        lambda package_name, target_version, this_version: f"{package_name} expected to be version {target_version}, but got {this_version} instead"
    )
    ambiguous_name_warning = (
        lambda package_name: f"Attempting to load the pkg '{package_name}', if you rather want to use the local module: use(use._ensure_path('{package_name}.py'))"
    )
    no_version_provided = (
        lambda: "No version was provided, even though auto_install was specified! Trying to load classically installed pkg instead."
    )
    classically_imported = (
        lambda name, this_version: f'Classically imported \'{name}\'. To pin this version: use("{name}", version="{this_version}")'
    )
    pebkac_missing_hash = (
        lambda name, package_name, version, hashes: f"""Failed to auto-install {package_name!r} because hashes is missing. This may work:
use("{name}", version="{version!s}", hashes={hashes!r}, modes=use.auto_install)"""
    )
    pebkac_unsupported = (
        lambda package_name: f"We could not find any version or release for {package_name} that could satisfy our requirements!"
    )
    pip_json_mess = (
        lambda package_name, target_version: f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI."
    )
    no_version_or_hash_provided = (
        lambda name, package_name, version, hashes: f"""Please specify version and hash for auto-installation of {package_name!r}.
To get some valuable insight on the health of this pkg, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version:
use("{name}", version="{version!s}", hashes={hashes!r}, modes=use.auto_install)"""
    )
    cant_import = (
        lambda name: f"No pkg installed named {name} and auto-installation not requested. Aborting."
    )
    cant_import_no_version = (
        lambda package_name: f"Failed to auto-install '{package_name}' because no version was specified."
    )
    venv_unavailable = (
        lambda python_exe, python_version, python_platform: f"""
Your system does not have a usable 'venv' pkg for this version of Python:
   Path =     {python_exe}
   Version =  {python_version}
   Platform = {python_platform}

Please run:
   sudo apt update
   sudo apt install python3-venv
to install the necessary packages.

You can test if your version of venv is working by running:
  {python_exe} -m venv testenv && ls -lA testenv/bin/python
"""
    )
    no_distribution_found = (
        lambda package_name, version: f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
    )


class StrMessage(Message):
    cant_import = (
        lambda package_name: f"No pkg installed named {package_name} and auto-installation not requested. Aborting."
    )


class TupleMessage(Message):
    pass


class KwargsMessage(Message):
    pass


#% Installation and Import Functions


@pipes
def _ensure_path(value: Union[bytes, str, furl.Path, Path]) -> Path:
    if isinstance(value, (str, bytes)):
        return Path(value).absolute()
    if isinstance(value, furl.Path):
        return (
            Path.cwd(),
            value.segments << map(Path) << tuple << reduce(Path.__truediv__),
        ) << reduce(Path.__truediv__)
    return value


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
        return (self.archive, [m.name for m in self.archive.getmembers() if m.type == b"0"])

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


@pipes
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
                            mod.__name__[len(parent_mod.__name__) + 1 :] + "/__init__.py"
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
                or isinstance(_ensure_loader(v), (SourceFileLoader, zipimport.zipimporter))
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
    func=None, *, name: str, **kwargs
) -> Union[ModuleType, Exception]:

    if func:
        result = func(name=name, **kwargs)
        if isinstance(result, ModuleType):
            return result
    return RuntimeWarning(Message.cant_import_no_version(name))


def _pebkac_version_no_hash(
    func=None, *, name: str, version: Version, hash_algo, package_name: str, message_formatter: Callable[[str, str, Version, set[str]], str]=Message.pebkac_missing_hash, **kwargs
) -> Union[Exception, ModuleType]:
    if func:
        result = func(**all_kwargs(_pebkac_version_no_hash, locals()))
        if isinstance(result, ModuleType):
            return result
    try:
        hashes = {
            entry.digests.get(hash_algo.name)
            for entry in (
                _get_package_data(package_name).releases[version]
            )
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


@pipes
def _pebkac_no_version_no_hash(*, name, package_name, hash_algo, **kwargs) -> Exception:
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
            message_formatter=Message.no_version_or_hash_provided
        )
        
    rw = RuntimeWarning(Message.pebkac_unsupported(package_name))
    rw.name = package_name
    return rw


def _import_public_no_install(
    *,
    package_name,
    module_name,
    spec,
    **kwargs,
) -> Union[ModuleType, Exception]:
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
    func=None,
    *,
    name,
    version,
    package_name,
    rest,
    hash_algo=Hash.sha256.name,
    self=None,
    **kwargs,
) -> Union[ModuleType, BaseException]:
    package_name, rest = _parse_name(name)

    if func:
        result = func(**all_kwargs(_auto_install, locals()))
        if isinstance(result, ModuleType):
            return result

    query = self.execute_wrapped(
        f"""
        SELECT 
            artifacts.id, import_relpath,
            artifact_path, installation_path, module_path, import_name
        FROM distributions
        JOIN artifacts ON artifacts.id = distributions.id
        WHERE name='{package_name}' AND version='{version!s}'
        ORDER BY artifacts.id DESC
        """
    ).fetchone()

    if not query or "installation_path" not in query:
        query = _find_or_install(package_name, version, force_install=True)
    
    artifact_path = _ensure_path(query["artifact_path"])
    module_path = _ensure_path(query["module_path"])
    
    # trying to import directly from zip
    _clean_sys_modules(rest)
    try:
        importer = zipimport.zipimporter(artifact_path)
        return importer.load_module(query["import_name"])
    except BaseException as zerr:
        traceback.print_exc()
    
    orig_cwd = Path.cwd()
    mod = None
    
    if not query or "installation_path" not in query:
        query = _find_or_install(package_name, version, force_install=True)
    
    assert "installation_path" in query
    assert query["installation_path"]
    installation_path = _ensure_path(query["installation_path"])
    try:
        os.chdir(installation_path)
        if "/" in name:
            import_name = name.split("/")[-1]
            for module_path in (
              installation_path / import_name.replace(".", "/") / "__init__.py",
              installation_path / (import_name.replace(".", "/") + ".py")
            ):
              if module_path.exists():
                break
        else:
            import_name = str(module_path.relative_to(installation_path)).replace("\\", "/").replace("/__init__.py", "").replace("-", "_")
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
                import_relpath=str(_ensure_path(module_path).relative_to(installation_path)),
                version=version,
                artifact_path=artifact_path,
                hash_value=hash_algo.value(artifact_path.read_bytes()).hexdigest(),
                module_path=module_path,
                installation_path=installation_path,
                import_name=import_name
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
    ret = None, None
    try:
        site_dirs = list(
            (
                sys.modules["use"].home / "venv" / package_name 
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
                    pps = [
                        pp for pp in dist.files if pp.as_posix() == relp
                    ]
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
        log.info("_find_module_in_venv(package_name=%s, version=%s, relp=%s) -> %s", repr(package_name), repr(version), repr(relp), repr(ret))
    
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
        info.update(_filtered_and_ordered_data(_get_package_data(package_name), version)[0].dict())
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
        log.info(f"pure python pkg: {package_name, version, use.home}")
        return out_info

    venv_root = _venv_root(package_name, version, use.home)
    out_info["installation_path"] = venv_root
    python_exe = Path(sys.executable)
    env = _get_venv_env(venv_root)
    module_paths = venv_root.rglob(f"**/{relp}")
    if force_install or not any(module_paths):
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
                        name=(
                          package_name + "/" + rest.replace("/", ".")),
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


@cache(maxsize=512, typed=True)
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


@pipes
def _filtered_and_ordered_data(data: PyPI_Project, version: Version = None) -> list[PyPI_Release]:
    if version:
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


@pipes
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
            if "VERBOSE" in os.environ: \
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


def _ensure_version(func, *, name, version, **kwargs) -> Union[ModuleType, Exception]:
    result = func(**all_kwargs(_ensure_version, locals()))
    if not isinstance(result, ModuleType):
        return result

    this_version = _get_version(mod=result)

    if this_version == version:
        return result
    else:
        return AmbiguityWarning(Message.version_warning(name, version, this_version))


def _fail_or_default(exception: BaseException, default: Any):
    if default is not mode.fastfail:
        return default  # TODO: write test for default
    else:
        raise exception


#% Main Classes


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

    def _set_up_files_and_directories(self):
        self.home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))
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
  "import_name" TEXT,
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
        import_name: str,
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
INSERT OR IGNORE INTO artifacts (distribution_id, import_relpath, artifact_path, module_path, import_name)
VALUES ({self.registry.lastrowid}, '{import_relpath}', '{artifact_path}', '{module_path}', '{import_name}')
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
                        _ensure_path(inspect.currentframe().f_back.f_back.f_code.co_filename)
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

    @__call__.register(type(None))  # singledispatch is picky - can't be anything but a type
    def _use_kwargs(
        self,
        _: None,  # sic! otherwise single-dispatch with 'empty' *args won't work
        /,
        *,
        name: str,
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
            name=name,
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
        log.debug(f"use-package: {name}, {package_name}, {module_name}, {version}, {hashes}")
        if isinstance(hashes, str):
            hashes = set([hashes])
        hashes = set(hashes) if hashes else set()
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
use.__dict__.update({k: v for k, v in globals().items()})  # to avoid recursion-confusion
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