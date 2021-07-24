"""
Basically, there are three types of package-installations:
* pure python code (which works anywhere with a python interpreter, no problem there)
* binary packages (which work only on the system for which they were built for)
* packages in other languages that need to be compiled on the target system

For the user, all of those appear equal and "pip-installable", but under the surface, they require very different handling.
Now, we've given up to handle all special cases in our main code, because it's a mess.
Instead, if anyone wants to do something special, they can do it in this file.

The basic idea is that use(name, version) with auto_install maps to a dict of commands that will be run to install the package. 
"""

import gzip
import importlib
import os
import sys
import tarfile
import tempfile
import traceback
import zipfile
from importlib.machinery import EXTENSION_SUFFIXES
from logging import DEBUG, StreamHandler, getLogger, root
from pathlib import Path

from packaging.specifiers import SpecifierSet

# this is possible because we don't *import* this file, but use() it!
import use

root.addHandler(StreamHandler(sys.stderr))
if "DEBUG" in os.environ: root.setLevel(DEBUG)
log = getLogger(__name__)

def readstring(path, lines=False, /, encoding="ISO-8859-1", 
    raw_lines=False):
    mode = "rb" if encoding is None else "r"
    with open(path, mode, buffering=-1,
        encoding=encoding, newline=("\x0a" if encoding else None)) as f:
        if lines:
            if raw_lines:
                return f.readlines()
            else:
                return list(map(str.rstrip, f.readlines()))
        return f.read()

def remove_cached_module(module_name):
    module_to_del = []
    module_parts = module_name.split(".")
    for part in module_parts:
        module_to_del.append(part)
        module_key = ".".join(module_to_del)
        if module_key in sys.modules:
            log.info("Deleting sys.modules[%s]",
                repr(module_key))
            del sys.modules[module_key]

def create_solib_links(archive: zipfile.ZipFile, folder: Path):
    log.debug(f"create_solib_links({archive=}, {folder=})")
    entries = archive.getnames() if hasattr(archive, "getnames") else archive.namelist()
    log.debug(f"archive {entries=}")
    solibs = [*filter(lambda f: any(map(f.endswith, EXTENSION_SUFFIXES)), entries)]
    if not solibs:
        log.debug(f"No solibs found in archive")
        return
    # Set up links from 'xyz.cpython-3#-<...>.so' to 'xyz.so'
    log.debug(f"Creating {len(solibs)} symlinks for extensions...")
    log.debug(f"solibs = {solibs}")
    for solib in solibs:
        sofile = folder / solib
        log.debug(f"{sofile=}, {folder=}, {solib=}")
        split_on = [".python", ".cpython", ".cp"]
        simple_name, os_ext = None, EXTENSION_SUFFIXES[-1]
        for s in split_on:
            if not s in sofile.name: continue
            simple_name = sofile.name.split(s)[0]
        if simple_name is None: continue
        link = Path(sofile.parent / f"{simple_name}{os_ext}")
        if link == sofile: continue
        log.debug(f"{link=}, {sofile=}")
        link.unlink(missing_ok=True)
        link.symlink_to(sofile)

def save_module_info(package_name, rdists, version, url, path, that_hash, folder):
    if package_name not in rdists:
        rdists[package_name] = {}
    if version not in rdists[package_name]:
        rdists[package_name][version] = {}
    # Update package version metadata
    assert url is not None
    rdists[package_name][version].update({
        "package": package_name,
        "version": version,
        "url": url.human_repr(),
        "path": str(path) if path else None,
        "folder": folder.absolute().as_uri(),
        "filename": path.name,
        "hash": that_hash
    })
    use.persist_registry()

def ensure_extracted(path, folder):
    if not folder.exists():
        folder.mkdir(mode=0o755, parents=True, exist_ok=True)
        log.info("Extracting %s (from %s) to %s ...", path, url, folder)
        fileobj = archive = None
        if path.suffix in (".whl", ".egg", ".zip"):
            fileobj = open(tempfile.mkstemp()[0], "w")
            archive = zipfile.ZipFile(path, "r")
        else:
            fileobj = (gzip.open if path.suffix == ".gz" else open)(path, "r")
            archive = tarfile.TarFile(fileobj=fileobj, mode="r")
        with archive as file:
            with fileobj as _:
                file.extractall(folder)
                create_solib_links(file, folder)

@use.register_hack("numpy", specifier=SpecifierSet(">=1.0"))
def numpy(*, package_name, rdists, version, url, path, that_hash, folder, fatal_exceptions, module_name):
    log.debug("hacking numpy!")
    ensure_extracted(path, folder)
    if sys.path[0] != "": sys.path.insert(0, "")
    original_cwd = Path.cwd()
    exc = mod = None
    try:
        os.chdir(folder)
        mod = importlib.import_module(module_name)
    except ImportError:
        if fatal_exceptions: raise
        exc = traceback.format_exc()
    finally:
        remove_cached_module(module_name)
        os.chdir(original_cwd)
    if not exc:
        save_module_info(package_name, rdists, version, url, path, that_hash, folder)
    return mod

@use.register_hack("protobuf", specifier=SpecifierSet(">=1.0"))
def protobuf(*, package_name, rdists, version, url, path, that_hash, folder, fatal_exceptions, module_name):
    log.debug("hacking protobuf!")
    ensure_extracted(path, folder)
    if sys.path[0] != "": sys.path.insert(0, "")
    original_cwd = Path.cwd()
    exc = mod = None
    try:
        os.chdir(folder)
        log.info("in dir: %s; original_cwd=%s", Path.cwd(), original_cwd)
        # this is needed because protobuf corrupts sys.path somehow into looking here
        tgt = use.home / Path(f".local/lib/python3.{sys.version_info[1]}/site-packages");
        log.info("folder=%s, symlink_to(tgt=%s)", folder, tgt)
        # remove any existing symlink here in case we already loaded a different version
        if tgt.exists(): tgt.unlink()
        tgt.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        tgt.symlink_to(folder.absolute())
        pth_src = "\n\n".join(readstring(str(pth_path)) for pth_path in folder.glob("*.pth"))
        log.info("pth_src=[%s]", pth_src)
        # this variable is needed because protobuf assumes it will be in the stack...
        sitedir = str(folder)
        log.info("sitedir=[%s]", sitedir)
        # compile the hacky duct-tape that protobuf embeds in its .pth file
        # to make their 'google' namespace work properly - disgusting
        exec(compile(pth_src, "pth_file.py", "exec"))
        if sys.path[0] != "": sys.path.insert(0, "")
        remove_cached_module(module_name)
        mod = use(Path("./google/protobuf/__init__.py"))
        sys.modules[module_name] = mod
        log.info("returning mod=%s", mod)
        mod = importlib.import_module(module_name)
    except ImportError:
        if fatal_exceptions: raise
        exc = traceback.format_exc()
    finally:
        remove_cached_module(module_name)
        os.chdir(original_cwd)
    if not exc:
        save_module_info(package_name, rdists, version, url, path, that_hash, folder)
    return mod