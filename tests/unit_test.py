import os
import sys

here = os.path.split(os.path.abspath(os.path.dirname(__file__)))
src = os.path.join(here[0], "src")
sys.path.insert(0,src)

import warnings
from pathlib import Path
from unittest import TestCase, skip

import pytest
import use
from yarl import URL


@pytest.fixture()
def reuse():
  # making a completely new one each time would take ages (_registry)
  use._using = {}
  use._aspects = {}
  use._reloaders = {}
  return use 

def test_other_case(reuse):
  with pytest.raises(NotImplementedError):
    reuse(2)

def test_fail_dir(reuse):
  with pytest.raises(ImportError):
    reuse(Path(""))
def test_simple_path(reuse):
    foo_path = Path(".tests/foo.py")
    print(f"loading foo module via use(Path('{foo_path}'))")
    mod = reuse(Path(foo_path), initial_globals={"a": 42})
    assert mod.test() == 42
  
def test_simple_url(reuse):
    import http.server
    port = 8089
    svr = http.server.HTTPServer(
      ("", port), http.server.SimpleHTTPRequestHandler
    )
    foo_uri = f"http://localhost:{port}/tests/.tests/foo.py"
    print(f"starting thread to handle HTTP request on port {port}")
    import threading
    thd = threading.Thread(target=svr.handle_request)
    thd.start()
    print(f"loading foo module via use(URL({foo_uri}))")
    with pytest.warns(use.NoValidationWarning):
      mod = reuse(URL(foo_uri), initial_globals={"a": 42})
      assert mod.test() == 42
    
def test_internet_url(reuse):
    foo_uri = "https://raw.githubusercontent.com/greyblue9/justuse/3f783e6781d810780a4bbd2a76efdee938dde704/tests/foo.py"
    print(f"loading foo module via use(URL({foo_uri}))")
    mod = reuse(
      URL(foo_uri), initial_globals={"a": 42},
      hash_algo=use.Hash.sha256, hash_value="b136efa1d0dab3caaeb68bc41258525533d9058aa925d3c0c5e98ca61200674d"
    )
    assert mod.test() == 42

def test_module_package_ambiguity(reuse):
  original_cwd = os.getcwd()
  os.chdir(Path("tests/.tests"))
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    reuse("sys")
    w_filtered = [*filter(
        lambda i: i.category is not DeprecationWarning, w)]
    assert len(w_filtered) == 1
    assert issubclass(w_filtered[-1].category, use.AmbiguityWarning)
    assert "local module" in str(w_filtered[-1].message)
  os.chdir(original_cwd)
    
def test_builtin():
  # must be the original use because loading builtins requires looking up _using, which mustn't be wiped for this reason 
  mod = use("sys")
  assert mod.path is sys.path

def test_classical_install(reuse):
 with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mod = reuse("pytest")
    assert mod is pytest
    assert issubclass(w[-1].category, use.AmbiguityWarning)

  
def test_autoinstall_PEBKAC(reuse):
  # auto-install requested, but no version or hash_value specified
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    reuse("pytest", auto_install=True)
    assert issubclass(w[-1].category, use.AmbiguityWarning)
  
  # forgot hash_value
  with pytest.raises(RuntimeWarning):
    reuse("pytest", auto_install=True, version=-1)
  
  # forgot version
  with pytest.raises(RuntimeWarning):
    reuse("pytest", auto_install=True, hash_value="asdf")
    
  # impossible version
  with pytest.raises(ImportError):
    reuse("pytest", auto_install=True, version=-1, hash_value="asdf")
  
  # non-existing package
  with pytest.raises(ImportError):
    reuse("-^df", auto_install=True, version="0.0.1", hash_value="asdf")
    
def test_version_warning(reuse):
  # no auto-install requested, wrong version only gives a warning
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    reuse("pytest", version=-1)
    assert issubclass(w[-1].category, use.VersionWarning)

def test_download_package(reuse):
  # https://pypi.org/project/example-pypi-package/
  reuse("example-pypi-package", version="0.1.0", hash_value="ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77", auto_install=True)