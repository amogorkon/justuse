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

def create_solib_links(archive: zipfile.ZipFile, folder: Path):
    log.debug(f"create_solib_links({archive=}, {folder=})")
    # EXTENSION_SUFFIXES  == ['.cpython-38-x86_64-linux-gnu.so', '.abi3.so', '.so'] or ['.cp39-win_amd64.pyd', '.pyd']
    entries = archive.getnames() if hasattr(archive, "getnames") \
        else archive.namelist()
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


@use.register_hack("numpy", specifier=SpecifierSet(">=1.0"))
def numpy(*, package_name, rdists, version, url, path, that_hash, folder, fatal_exceptions, module_name):
    print("hacking numpy!")
    log.debug(f"outside of create_solib_links(...)")
    if package_name not in rdists:
        rdists[package_name] = {}
    if version not in rdists[package_name]:
        rdists[package_name][version] = {}
    # Update package version metadata
    assert url is not None
    mod = None
    rdist_info = rdists[package_name][version]
    rdist_info.update({
        "package": package_name,
        "version": version,
        "url": url.human_repr(),
        "path": str(path) if path else None,
        "folder": folder.absolute().as_uri(),
        "filename": path.name,
        "hash": that_hash
    })
    use.persist_registry()
    
    if not folder.exists():
        folder.mkdir(mode=0o755, parents=True, exist_ok=True)
        print("Extracting to", folder, "...")

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
        print("Extracted.")
    original_cwd = Path.cwd()
    
    os.chdir(folder)
    exc = None
    importlib.invalidate_caches()
    if sys.path[0] != "":
        sys.path.insert(0, "")
    try:
        log.debug("Trying importlib.import_module")
        log.debug("  with cwd=%s,", os.getcwd())
        log.debug("  sys.path=%s", sys.path)
        mod = importlib.import_module(module_name)
    except ImportError:
        if fatal_exceptions: raise
        exc = traceback.format_exc()
    finally:
        module_to_del = []
        module_parts = module_name.split(".")
        for part in module_parts:
            module_to_del.append(part)
            module_key = ".".join(module_to_del)
            if module_key in sys.modules:
                log.info("Deleting sys.modules[%s]",
                    repr(module_key))
                del sys.modules[module_key]
            

    for key in ("__name__", "__package__", "__path__", "__file__", "__version__", "__author__"):
        if not hasattr(mod, key): continue
        rdist_info[key] = getattr(mod, key)
    if not exc:
        print(f"Successfully loaded {package_name}, version {version}.")
    os.chdir(original_cwd)
    return mod

@use.register_hack("protobuf", specifier=SpecifierSet(">=1.0"))
def protobuf(*, package_name, rdists, version, url, path, that_hash, folder, fatal_exceptions, module_name):
  original_cwd = Path.cwd()

  # Update package version metadata
  assert url is not None
  mod = None
  if not folder.exists():
      folder.mkdir(mode=0o755, parents=True, exist_ok=True)
      print("Extracting to", folder, "...")

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
      log.info("Extracted.")
  log.info("PROTOBUF: in dir: %s; original_cwd=%s", Path.cwd(), original_cwd)
  tgt = use.home / Path(f".local/lib/python3.{sys.version_info[1]}/site-packages");
  log.info("folder=%s, symlink_to(tgt=%s)", folder, tgt)
  if tgt.exists():
      tgt.unlink()
  tgt.parent.mkdir(mode=0o755, parents=True, exist_ok=True)

  log.info("PROTOBUF: tgt=%s, symlink_to(folder=%s)", tgt, folder)
  tgt.symlink_to(folder.absolute())
  log.info("SUCCEEDED: tgt=%s, symlink_to(folder=%s)", tgt, folder)
  os.chdir(str(folder))

  pwd = Path.cwd()
  log.info("PROTOBUF: Entered directory: pwd=%s", pwd)
  pth_src = "\n\n".join(
      readstring(str(pth_path)) for pth_path in folder.glob("*.pth"))
  log.info("pth_src=[%s]", pth_src)
  sitedir = str(folder)
  log.info("sitedir=[%s]", sitedir)
  rslt = exec(
    compile(
      pth_src,
      "pth_file.py",
      "exec"
    ),
  )
  log.info("rslt = %s", rslt)

  exc = None
  importlib.invalidate_caches()
  if sys.path[0] != "":
      sys.path.insert(0, "")
  try:
    module_to_del = []
    module_parts = module_name.split(".")
    for part in module_parts:
        module_to_del.append(part)
        module_key = ".".join(module_to_del)
        if module_key in sys.modules:
            log.info("Deleting sys.modules[%s]",
                repr(module_key))
            del sys.modules[module_key]


    if not mod:
      log.debug("Trying importlib.import_module")
      log.debug("  with cwd=%s,", os.getcwd())
      log.debug("  sys.path=%s", sys.path)
      log.debug("  sys.modules=%s", sys.modules)

      #mod_goog = use(Path("./google/__init__.py"))
      #log.debug("  mod_goog=%s", mod_goog)
      #log.debug("mod_goog.__spec__=%s",
      #   getattr(mod_goog,"__spec__",""))

      #mod_goog.__package__ = "google"
      #mod_goog.__name__ = "__init__"
      #sys.modules["google"] = mod_goog

      mod_pbuf = use(Path("./google/protobuf/__init__.py"))
      log.debug("  mod_pbuf=%s", mod_pbuf)
      log.debug("mod_pbuf.__spec__=%s",
         getattr(mod_pbuf,"__spec__",""))

      sys.modules["google.protobuf"] = mod_pbuf

      # setattr(mod_goog, "protobuf", mod_pbuf)
      mod = mod_pbuf

      log.debug("  mod=%s", mod)
      log.debug("mod.__file__=%s",getattr(mod, "__file__",""))
      log.debug("mod.__spec__=%s",getattr(mod, "__spec__",""))
      log.debug("mod.__version__=%s",getattr(mod, "__version__",""))

  except BaseException as exc:
      log.error(exc)
      raise
  log.info("returning mod=%s", mod)

  if package_name not in rdists:
      rdists[package_name] = {}
  if version not in rdists[package_name]:
      rdists[package_name][version] = {}
  rdist_info = rdists[package_name][version]
  rdist_info.update({
      "package": package_name,
      "version": version,
      "url": url.human_repr(),
      "path": str(path) if path else None,
      "folder": folder.absolute().as_uri(),
      "filename": path.name,
      "hash": that_hash
  })
  use.persist_registry()
  return mod
