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
>>> tools = use(use._ensure_path("/media/sf_Dropbox/code/tools.py"), reloading=True)

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
import shlex
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
from functools import reduce, singledispatch, update_wrapper
from importlib import metadata
from importlib.machinery import SourceFileLoader
from itertools import chain, takewhile
from logging import DEBUG, INFO, NOTSET, WARN, StreamHandler, getLogger, root
from pathlib import Path
from pprint import pformat
from subprocess import PIPE, run
from textwrap import dedent
from types import ModuleType
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple, Union
from warnings import warn

import furl
import packaging
import requests
import toml
from furl import furl as URL
from packaging import tags
from packaging.specifiers import SpecifierSet
from packaging.version import Version as PkgVersion
from pip._internal.utils import compatibility_tags

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
    def __new__(cls, *args, **kwargs):
        if not args or not isinstance(args[0], Version):
            return super(cls, Version).__new__(cls)
        return args[0]

    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if isinstance(versionstr, Version):
            return

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


class PlatformTag(namedtuple("PlatformTag", ["platform"])):
    def __str__(self):
        return self.platform

    def __repr__(self):
        return self.platform


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
    use_version_warning = (
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
        lambda name: f"Attempting to load the package '{name}', if you rather want to use the local module: use(use._ensure_path('{name}.py'))"
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
        lambda name, package_name, version, hash_value: f"""Please specify version and hash for auto-installation of '{package_name}'.
To get some valuable insight on the health of this package, please check out https://snyk.io/advisor/python/{package_name}
If you want to auto-install the latest version: use("{name}", version="{version}", hashes={set([hash_value])}, modes=use.auto_install)
"""
    )
    cant_import = (
        lambda name: f"No package installed named {name} and auto-installation not requested. Aborting."
    )
    cant_import_no_version = (
        lambda package_name: f"Failed to auto-install '{package_name}' because no version was specified."
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


def _ensure_path(value_for_path):
    from pathlib import Path

    if value_for_path is None:
        raise BaseException("value_for_path is None")
    if value_for_path in ("", b""):
        raise BaseException("value_for_path is empty string or bytes")
    if isinstance(value_for_path, Path):
        return value_for_path
    if isinstance(value_for_path, (str, bytes)):
        return Path(value_for_path)
    if isinstance(value_for_path, furl.Path):
        return reduce(Path.__truediv__, value_for_path.segments, Path.cwd())
    raise TypeError(f"Attempt to create a Path from an illegal value: {value_for_path!r}")


@cache
def get_supported() -> FrozenSet[PlatformTag]:
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, package_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    items: List[PlatformTag] = []

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
                    _ensure_path(n).stem,
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
            rel3 = _ensure_path(rel2)
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


def _venv_root(package_name, version, home) -> Path:
    venv_root = home / "venv" / package_name / (str(version) or "0.0.0")
    if not venv_root.exists():
        venv_root.mkdir(parents=True)
    return venv_root


def _venv_is_win() -> bool:
    return sys.platform.lower().startswith("win")


def _pebkac_no_version_hash(func=None, *, name: str, **kwargs) -> "ModuleType|Exception":
    if func:
        result = func(name=name, **kwargs)
        if isinstance(result, ModuleType):
            return result
    return RuntimeWarning(Message.cant_import_no_version(name))


def _pebkac_version_no_hash(
    func=None, *, package_name, version, hash_algo, **kwargs
) -> "Exception | ModuleType":
    if func:
        result = func(package_name=package_name, version=version, **kwargs)
        if isinstance(result, ModuleType):
            return result

    data = _get_filtered_data(_get_package_data(package_name))
    entry = data["releases"][str(version)][-1]
    that_hash = entry["digests"][hash_algo.name]
    if that_hash:
        return RuntimeWarning(
            Message.pebkac_missing_hash(
                package_name,
                version,
                that_hash,
            )
        )
    return RuntimeWarning(
        f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
    )


def _pebkac_no_version_no_hash(*, name, package_name, hash_algo, **kwargs) -> Exception:
    # let's try to make an educated guess and give a useful suggestion
    data = _get_filtered_data(_get_package_data(package_name))
    for version, infos in sorted(data["releases"].items()):
        hash_value = infos[0]["digests"][hash_algo.name]
        if hash_value:
            return RuntimeWarning(
                Message.no_version_or_hash_provided(
                    name,
                    package_name,
                    version,
                    hash_value,
                )
            )

    return RuntimeWarning(Message.pebkac_unsupported(package_name))


def _import_public_no_install(
    *,
    name,
    spec,
    aspectize,
    fatal_exceptions=False,
    **kwargs,
) -> "ModuleType|Exception":
    # builtin?
    package_name, rest = _parse_name(name)
    builtin = False
    try:
        metadata.PathDistribution.from_name(package_name)
    except metadata.PackageNotFoundError:  # indeed builtin!
        builtin = True

    if builtin:
        mod = spec.loader.create_module(spec)
        spec.loader.exec_module(mod)  # ! => cache
        if aspectize:
            warn(Message.aspectize_builtins_warning(), RuntimeWarning)
        return mod

    # it seems to be installed in some way, for instance via pip
    return importlib.import_module(rest)  # ! => cache


def _parse_name(name):
    match = re.match(r"(?P<package_name>[^.]+)\.?(?P<rest>[a-zA-Z0-9._]+)?", name)
    assert match, f"Invalid name spec: {name!r}"
    names = match.groupdict()
    return names["package_name"], names["rest"] or name


def _auto_install(
    func=None,
    *,
    name,
    hash_algo,
    version,
    **kwargs,
):
    package_name, rest = _parse_name(name)

    if func:
        result = func(**all_kwargs(_auto_install, locals()))

    query = use.registry.execute(
        f"SELECT id, installation_path FROM distributions WHERE name='{package_name}' AND version='{str(version)}'",
    ).fetchone()

    if query:
        installation_path = query["installation_path"]
        query = use.registry.execute(
            "SELECT path FROM artifacts WHERE distribution_id=?",
            [
                query["id"],
            ],
        ).fetchone()
    path = None
    if query:
        path = _ensure_path(query["path"])
    if path:
        _update_hashes(**all_kwargs(_auto_install, locals()))

    if path:
        # trying to import directly from zip
        try:
            importer = zipimport.zipimporter(path)
            mod = importer.load_module(rest)
            print("Direct zipimport of", rest, "successful.")
        except (zipimport.ZipImportError, ImportError):
            if config["debugging"]:
                log.debug(traceback.format_exc())
            print("Direct zipimport failed, attempting to extract and load manually...")

    installation_path = (path.parent / path.stem) if path else None

    out_info = {}
    mod = _load_venv_mod(
        name=name,
        version=version,
        artifact_path=path,
        out_info=out_info,
    )
    path = path or out_info["artifact_path"]
    installation_path = out_info["installation_path"]
    this_version = Version(out_info["version"])
    that_hash = out_info["digests"][hash_algo.name]

    use._save_module_info(
        name=package_name,
        version=this_version,
        artifact_path=path,
        hash_value=that_hash,
        installation_path=installation_path,
    )
    return mod


def _update_hashes(*, package_name, version, default, hash_algo, hashes, **kwargs) -> None:
    found = None
    try:
        data = _get_filtered_data(_get_package_data(package_name))
        infos = data["releases"][str(version)]
        all_hashes = set()
        for entry in infos:
            url = URL(entry["url"])
            log.error("url = %s", url)
            log.debug(f"looking at {entry=}")
            all_hashes.add(that_hash := entry["digests"].get(hash_algo.name))
            if hashes.intersection(all_hashes):
                found = (entry, that_hash)
                hit = (version, that_hash)
                log.info(f"Matches user hash: {entry=} {hit=}")
                break
        if not found:
            return AutoInstallationError(
                f"Tried to auto-install ({package_name=!r}) with {version=!r} but failed because none of the available hashes ({all_hashes=!r}) match the expected hash ({hashes=!r})."
            )
        entry, that_hash = found
        if that_hash is not None:
            assert isinstance(hashes, set)
            hashes.add(that_hash)
    except KeyError as be:  # json issuesa
        msg = f"request to https://pypi.org/pypi/{package_name}/{version}/json lead to an error: {be}"
        return RuntimeError(msg)


# these are aliases for the purpose of aspectizing modules


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
                encoding="UTF-8",
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
                "%s\n1b[0m",
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


def _find_version(package_name, version=None):
    data = _get_filtered_data(_get_package_data(package_name))
    if not version or str(version) in ("0.0.0", "None"):
        version, infos = [*data["releases"].items()][-1]
        return infos[-1]
    for rel_version, infos in data["releases"].items():
        if str(rel_version) == str(version):
            return infos[-1]
    assert False, f"No match found for {package_name=!r}, {version=!r}"


def _find_exe(venv_root):
    env = {
        "PATH": str(venv_root / "bin")
        + os.path.pathsep
        + str(venv_root / "Scripts")
        + os.path.pathsep
        + os.environ["PATH"],
    }
    for p in (
        *venv_root.rglob("**/bin/python"),
        *venv_root.rglob("**/bin/python.exe"),
        *venv_root.rglob("**/Scripts/python"),
        *venv_root.rglob("**/Scripts/python*.exe"),
    ):
        return p, env, _ensure_path(p).parent, _ensure_path(p).name
    o = _process(sys.executable, "-m", "venv", venv_root)
    for p in (
        *venv_root.rglob("**/bin/python"),
        *venv_root.rglob("**/bin/python.exe"),
        *venv_root.rglob("**/Scripts/python"),
        *venv_root.rglob("**/Scripts/python*.exe"),
    ):
        return p, env, _ensure_path(p).parent, _ensure_path(p).name


def _download_artifact(url, artifact_path):
    artifact_path.write_bytes(requests.get(url).content)


def _load_venv_mod(name, version=None, artifact_path=None, url=None, out_info=None) -> ModuleType:
    if "None" in str(artifact_path):
        artifact_path = None
    
    log.error("_load_venv_mod(name=%s, version=%s, artifact_path=%s, out_info=%s)", name, version, artifact_path, out_info)
    if not version or str(version) in ("0.0.0", "None", ""):
        log.error("resetting version")
        version = None
    log.error("_load_venv_mod(name=%s, version=%s, artifact_path=%s, out_info=%s)", name, version, artifact_path, out_info)
    
    info = (out_info := (out_info if out_info is not None else {}))
    package_name, rest = _parse_name(name)
    
    log.error("calling _find_version(package_name=%s, version=%s)", package_name, version)
    vers_info = _find_version(package_name, version)
    log.error("_find_version(packagw_name=%s, version=%s) returned %s", package_name, version, vers_info)
    version = Version(vers_info["version"])
    log.error("versoion = %s", version)
    info.update(vers_info)
    url = URL(info["url"])
    filename = info["filename"]
    artifact_path = sys.modules["use"].home / "packages" / info["filename"]
    
    
    log.error("_load_venv_mod(name=%s, version=%s, artifact_path=%s, out_info=%s)", name, version, artifact_path, out_info)
    
    venv_root = _venv_root(package_name, version, use.home)
    p, env, venv_bin, python_exe = _find_exe(venv_root)
    
    install_item = package_name + "==" + str(version)
    if url and not artifact_path.exists():
        _download_artifact(url, artifact_path)
    if artifact_path.exists():
        install_item = artifact_path
    log.info("Installing %s using pip", install_item)
    
    log.error("%s\n", pformat(locals()))
    output = _process(
        str(venv_bin / python_exe),
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-color",
        "install",
        "--pre",
        "-v",
        "-v",
        "-v",
        "--prefer-binary",
        "--exists-action",
        "i",
        "--no-build-isolation",
        "--no-cache-dir",
        "--no-compile",
        "--no-warn-script-location",
        "--no-warn-conflicts",
        install_item,
        env=env,
    )
    match = re.compile(
        "Added (?P<package_name>[^= ]+)(?:==(?P<version>[^ ]+)|) "
        "from (?P<url>[^# ,\n]+(?:/(?P<filename>[^#, \n/]+)))"
        "(?:#(\\S*)|)(?=\\s)",
        re.DOTALL,
    ).search("%s\n\n%s" % (output.stdout, output.stderr))
    info.update(match.groupdict() if match else {})
    orig_cwd = Path.cwd()
    meta = archive_meta(artifact_path)
    meta.update(info)
    relp = meta["import_relpath"]
    name_segments = ".".join(relp.split(".")[0:-1]).split("-")[0].replace("/", ".")
    package_name, rest = _parse_name(name_segments)
    module_paths = [*venv_root.rglob(f"**/{relp}")]
    for module_path in module_paths:
        installation_path = module_path
        for _ in range(len(name_segments.split("."))):
            installation_path = installation_path.parent
        try:
            log.error("installation_path = %s", installation_path)
            os.chdir(str(installation_path))
            out_info.update(
                {
                    "artifact_path": artifact_path,
                    "installation_path": installation_path,
                    "module_path": module_path,
                    "package_name": package_name,
                    "url": url,
                    "version": version,
                    "info": info,
                    **meta,
                    **info,
                }
            )
            return _load_venv_entry(name, module_path=module_path)
        finally:
            os.chdir(orig_cwd)
    assert (
        False
    ), f"{meta['import_relpath']=}, {relp=}, {name_segments=}, {package_name=}, {rest=}"


def _load_venv_entry(name, module_path) -> ModuleType:
    package_name, rest = _parse_name(name)
    _clean_sys_modules(name)
    log.info(
        "load_venv_entry package_name=%s rest=%s module_path=%s",
        package_name,
        rest,
        module_path,
    )
    with open(module_path, "rb") as code_file:
        return _build_mod(
            name=rest,
            code=code_file.read(),
            module_path=_ensure_path(module_path),
            initial_globals={},
            aspectize={},
        )


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
) -> ModuleType:

    package_name, rest = _parse_name(name)
    mod = None
    if "__init__" in module_path.stem:
        mod = ModuleType(rest + ".__init__")
    else:
        mod = ModuleType(rest)

    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__package__ = rest
    mod.__name__ = rest
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
    except KeyError:
        try:
            exec(compile(code, module_path, "exec"), vars(mod))
        except KeyError:  # reraise anything without handling - clean and simple.
            raise
    for (check, pattern), decorator in aspectize.items():
        _apply_aspect(mod, check, pattern, decorator, aspectize_dunders=aspectize_dunders)
    return mod


def _ensure_proxy(mod) -> ProxyModule:
    if mod.__class__ is not ModuleType:
        return mod
    return ProxyModule(mod)


def _ensure_version(func, *, name, version, **kwargs) -> "ModuleType | Exception":
    result = func(**all_kwargs(_ensure_version, locals()))
    if not isinstance(result, ModuleType):
        return result

    this_version = _get_version(mod=result)

    if this_version == version:
        return result
    else:
        return AmbiguityWarning(Message.version_warning(name, version, this_version))


def _fail_or_default(exception, default):
    assert isinstance(
        exception, BaseException
    ), f"_fail_or_default MUST be called with a valid exception, but got {exception=}, {default=}"
    if default is not mode.fastfail:
        return default  # TODO: write test for default
    else:
        raise exception


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
                            aspectize=self.aspectize,
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
        self.registry.row_factory = lambda cursor, row: {
            col[0]: row[idx] for idx, col in enumerate(cursor.description)
        }

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
            except KeyError:
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
            if not _ensure_path(path).exists():
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
                    UnexpectedHash(
                        f"{this_hash} does not match the expected hash {hash_value} - aborting!"
                    ),
                    default,
                )
        else:
            warn(Message.no_validation(url, hash_algo, this_hash), NoValidationWarning)
        name = str(url)
        try:
            mod = _build_mod(
                name=name,
                code=response.content,
                module_path=_ensure_path(url.path),
                initial_globals=initial_globals,
                aspectize=aspectize,
                aspectize_dunders=bool(Use.aspectize_dunders & modes),
            )
        except KeyError:
            raise
        if exc:
            return _fail_or_default(ImportError(exc), default)

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
            return _fail_or_default(ImportError(f"Can't import directory {path}"), default)

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
                            _ensure_path(
                                inspect.currentframe().f_back.f_back.f_code.co_filename
                            )
                            .resolve()
                            .parent
                        )
            if source_dir is None:
                source_dir = _ensure_path(main_mod.__loader__.path).parent
            if not source_dir.joinpath(path).exists():
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
                        aspectize=aspectize,
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
                        module_path=path,
                        aspectize=aspectize,
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
            assert isinstance(
                as_import, str
            ), f"as_import must be the name (as str) of the module as which it should be imported, got {as_import} ({type(as_import)}) instead."
            sys.modules[as_import] = mod
        frame = inspect.getframeinfo(inspect.currentframe())
        self._set_mod(name=name, mod=mod, frame=frame)
        return _ensure_proxy(mod)

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
        mod = None
        log.debug(f"use-str: {name} {version} {hashes}")
        package_name, rest = _parse_name(name)
        log.debug(f"use-str: {package_name}, {rest} {version} {hashes}")

        if isinstance(hashes, str):
            hashes = set([hashes])
        hashes = set(hashes) if hashes else set()

        # we use boolean flags to reduce the complexity of the call signature
        fatal_exceptions = bool(Use.fatal_exceptions & modes)
        auto_install = bool(Use.auto_install & modes)
        aspectize_dunders = bool(Use.aspectize_dunders & modes)
        aspectize = aspectize or {}

        version = (
            version
            if isinstance(version, Version)
            else (Version(version) if version else None)
        )

        # The "try and guess" behaviour is due to how classical imports work,
        # which is inherently ambiguous, but can't really be avoided for packages.
        # let's first see if the user might mean something else entirely
        if any(_ensure_path(".").glob(f"**/{rest}.py")):
            warn(Message.ambiguous_name_warning(name), AmbiguityWarning)
        spec = None

        if name in self._using:
            spec = self._using[name].spec
        elif not auto_install:
            spec = importlib.util.find_spec(name)

        # welcome to the buffet table, where everything is a lie
        # fmt: off
        case = (bool(version), bool(hashes), bool(spec), bool(auto_install))
        log.info("case = %s", case)
        result = {
            (False, False, False, False): lambda **kwargs: ImportError(Message.cant_import(name)),
            (False, False, False, True): _pebkac_no_version_no_hash,
            (False, False, True, False): _import_public_no_install,
            (False, True, False, False): lambda **kwargs: ImportError(Message.cant_import(name)),
            (True, False, False, False): lambda **kwargs: ImportError(Message.cant_import(name)),
            (False, False, True, True): lambda **kwargs: _auto_install(_import_public_no_install, **kwargs),
            (False, True, True, False): _import_public_no_install,
            (True, True, False, False): lambda **kwargs: ImportError(Message.cant_import(name)),
            (True, False, False, True): _pebkac_version_no_hash,
            (True, False, True, False): lambda **kwargs: _ensure_version(_import_public_no_install, **kwargs),
            (False, True, False, True): _pebkac_no_version_hash,
            (False, True, True, True): lambda **kwargs: _pebkac_no_version_hash(_import_public_no_install, **kwargs),
            (True, False, True, True): lambda **kwargs: _pebkac_version_no_hash(_ensure_version(_import_public_no_install, **kwargs), **kwargs),
            (True, True, False, True): _auto_install,
            (True, True, True, False): lambda **kwargs: _ensure_version(_import_public_no_install, **kwargs),
            (True, True, True, True): lambda **kwargs: _auto_install(_ensure_version(_import_public_no_install, **kwargs), **kwargs),
        }[case](**locals())
        log.info("result = %s", result)
        # fmt: on
        mod = result if isinstance(result, ModuleType) else mod
        exc = result if isinstance(result, BaseException) else None
        assert mod != None or exc != None

        if mod:
            for (check, pattern), decorator in aspectize.items():
                _apply_aspect(
                    mod, check, pattern, decorator, aspectize_dunders=aspectize_dunders
                )
            frame = inspect.getframeinfo(inspect.currentframe())
            self._set_mod(name=name, mod=mod, spec=spec, frame=frame)
            return _ensure_proxy(mod) or mod
        return _fail_or_default(result, default)


use = Use()
use.__dict__.update(globals())
if not test_version:
    sys.modules["use"] = use