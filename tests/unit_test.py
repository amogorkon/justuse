import os
import sys
from pathlib import Path

if Path("use").is_dir(): os.chdir("..")
import_base = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(import_base))

import re
import warnings
from pathlib import Path
from unittest import skip
from unittest.mock import patch

import pytest
import requests
import use
from yarl import URL


@pytest.fixture()
def reuse():
  # making a completely new one each time would take ages (_registry)
  use._using = {}
  use._aspects = {}
  use._reloaders = {}
  return use 

def get_sample_data():
    return requests.get(
    "https://raw.githubusercontent.com/greyblue9"
    "/junk/master/rels.json"
    ).json()

def test_access_to_home():
  test = use.Path.home() / ".justuse-python/packages/test"
  test.touch(mode=0o644, exist_ok=True)
  with open(test, "w") as file:
    file.write("test")
  assert test.exists()
  test.unlink()
  assert not test.exists()

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
    with http.server.HTTPServer(
      ("", port), http.server.SimpleHTTPRequestHandler
    ) as svr:
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
  with pytest.raises(RuntimeWarning):
    reuse("pytest", auto_install=True)
  
  # forgot hash_value
  with pytest.raises(AssertionError):
    reuse("pytest", auto_install=True, version="-1")
  
  # forgot version
  with pytest.raises(RuntimeWarning):
    reuse("pytest", auto_install=True, hash_value="asdf")
    
  # impossible version
  with pytest.raises(AssertionError):  # version must be either str or tuple
    reuse("pytest", auto_install=True, version=-1, hash_value="asdf")
  
  # non-existing package
  with pytest.raises(ImportError):
    reuse("4-^df", auto_install=True, version="0.0.1", hash_value="asdf")
    
def test_version_warning(reuse):
  # no auto-install requested, wrong version only gives a warning
  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    reuse("pytest", version="0.0")
    assert issubclass(w[-1].category, (use.AmbiguityWarning, use.VersionWarning))

def test_pure_python_package(reuse):
  # https://pypi.org/project/example-pypi-package/
  file = use.Path.home() / f".justuse-python/packages/example_pypi_package-0.1.0-py3-none-any.whl"
  file.unlink(missing_ok=True)
  test = reuse("example-pypi-package.examplepy", version="0.1.0", hash_value="ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77", auto_install=True)
  assert str(test.Number(2)) == "2"
  file.unlink()

@skip
def test_auto_install_native():
  use._registry = None
  use._registry = use.load_registry()
  with patch('use.config', {"debugging": True}, spec=True):  # ? not sure about that
    rw = None
    try:
      use("numpy", auto_install=True)
    except RuntimeWarning as w:
      rw = w
    assert rw, "Expected a RuntimeWarning from unversioned auto-install"
    match:Optional[re.Match] = re.search(
      "use\\("
        "\"(?P<name>.*)\", "
        "version=\"(?P<version>.*)\", "
        "hash_value=\"(?P<hash_value>.*)\", "
        "auto_install=True"
      "\\)",
      rw.args[0]
    )
    assert match, f"Format did not match regex: {rw.args[0]!r}"
    params:dict = match.groupdict()
    name = "numpy"
    version = params["version"]
    hash_value = params["hash_value"]
    print(f"calling use({name!r}, {params}, auto_install=True) ...")
    mod = use(name, hash_value=hash_value, version=version, auto_install=True)
    print(f"mod={mod}")
    assert mod, "No module was returned"
    assert mod.ndarray, "Wrong module was returned (expected 'nparray')"
    assert mod.__version__ == params["version"], "Wrong numpy version"
