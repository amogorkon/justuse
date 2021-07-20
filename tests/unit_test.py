import json
import os
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import packaging.tags
import packaging.version
import pytest
import requests
from yarl import URL

if Path("use").is_dir(): os.chdir("..")
import_base = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(import_base))
import use

__package__ = "tests"

@pytest.fixture()
def reuse():
    # making a completely new one each time would take ages (_registry)
    use._using = {}
    use._aspects = {}
    use._reloaders = {}
    return use 

def test_access_to_home(reuse):
    test = reuse.Path.home() / ".justuse-python/packages/test"
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
        reuse("pytest", modes=reuse.auto_install)
  
    # forgot hash_value
    with pytest.raises(packaging.version.InvalidVersion):
        reuse("pytest", modes=reuse.auto_install, version="-1")
  
    # forgot version
    with pytest.raises(RuntimeWarning):
        reuse("pytest", modes=reuse.auto_install, hash_value="asdf")
    
    # impossible version
    with pytest.raises(AssertionError):  # version must be either str or tuple
        reuse("pytest", modes=reuse.auto_install, version=-1, hash_value="asdf")
  
    # non-existing package
    with pytest.raises(ImportError):
        reuse("4-^df", modes=reuse.auto_install, version="0.0.1", hash_value="asdf")
    
def test_version_warning(reuse):
    # no auto-install requested, wrong version only gives a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        reuse("pytest", version="0.0")
        assert issubclass(w[-1].category, (use.AmbiguityWarning, use.VersionWarning))

def test_pure_python_package(reuse):
    # https://pypi.org/project/example-pypi-package/
    file = reuse.Path.home() / f".justuse-python/packages/example_pypi_package-0.1.0-py3-none-any.whl"
    file.unlink(missing_ok=True)
    test = reuse("example-pypi-package.examplepy", version="0.1.0", hash_value="ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77", modes=reuse.auto_install)
    assert str(test.Number(2)) == "2"
    file.unlink()

@pytest.mark.skipif(sys.platform.startswith("win") 
  and list(sys.version_info)[0:2] >= [3, 10], 
    reason="windows Auto-installing native modules is not supported")
def test_auto_install_native(reuse):
    rw = None
    try:
        use("protobuf", modes=reuse.auto_install, package_name="protobuf",
            module_name="google.protobuf")
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
        version = params["version"]
        hash_value = params["hash_value"]
        print(f"calling use({params}, auto_install=True) ...")
        mod = use("protobuf", version=version, hash_value=hash_value, modes=reuse.auto_install, package_name="protobuf", module_name="google.protobuf")
        print(f"mod={mod}")
        assert mod, "No module was returned"
        assert mod.__version__ == version

def test_registry_first_line_warning(reuse):
    with open(reuse.home / "registry.json") as file:
        assert file.readlines()[0].startswith("### WARNING")

def test_use_global_install(reuse):
    import foo
    with pytest.raises(NameError):
        foo.bar()
        
    reuse.install()
    assert foo.bar()
    reuse.uninstall()
    del foo

def test_registry(reuse):
    name, vers, hash_value = (
    "example-pypi-package.examplepy", "0.1.0",
    "ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77"
    )
    package_name, _ = name.split(".")
    file = use.Path.home() / f".justuse-python" / "packages" \
        / f"{package_name.replace('-','_')}-0.1.0-py3-none-any.whl"
    file.unlink(missing_ok=True)
    _ = reuse(name, version=vers, hash_value=hash_value, modes=reuse.auto_install)
    with open(Path.home() / ".justuse-python" / "registry.json", "rb") \
                as jsonfile:
        _extracted_from_test_registry_13(jsonfile, package_name, vers, file)

def _extracted_from_test_registry_13(jsonfile, package_name, vers, file):
    jsonbytes = jsonfile.read()
    jsondata = json.loads(b"\x0a".join(
    [*filter(None,
        filter(lambda i: not i.startswith(b"#"),
        jsonbytes.splitlines()))]
    ))
    assert jsondata, "An empty registry was written to disk."
    dists = jsondata["distributions"]
    assert dists, "No distribution metadata saved to registry."
    package_dists = dists[package_name]
    assert package_dists, \
        f"No distribution metadata saved for package {package_name}"
    dist = package_dists[vers]
    assert dist, "No distribution saved for the expected version."
    assert "path" in dist, "Registry metadata contains no 'path'."
    path = Path(dist["path"])
    assert path.exists(), f"The package {path} did not get written."
    assert path.absolute() == file.absolute(), \
        f"The package did not get written to the expected location."
    for k, v in use._registry.items():
        jsonv = jsondata.get(k, ...)
        if isinstance(jsonv, dict) and isinstance(v, defaultdict):
            v = dict(v)
        assert jsonv == v, \
            f"The registry does not match the persisted json" \
            f" for key '{k}': expected: {jsonv}, found: {v}"

def test_is_version_satisfied(reuse):
    sys_version = packaging.version.Version("3.6.0")
    # google.protobuf 1.19.5 normal case
    info = {'comment_text': '', 'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 'downloads': -1, 'filename': 'google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'has_sig': False, 'md5_digest': '2651049b70d2ec07d8afd7637f198807', 'packagetype': 'bdist_wheel', 'python_version': 'cp36', 'requires_python': '>=3.6', 'size': 15599590, 'upload_time': '2021-01-05T17:19:38', 'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'yanked': False, 'yanked_reason': None}
    assert reuse._is_version_satisfied(info, sys_version)
    
    # requires >= python 4!
    info = {'comment_text': '', 'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 'downloads': -1, 'filename': 'google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'has_sig': False, 'md5_digest': '2651049b70d2ec07d8afd7637f198807', 'packagetype': 'bdist_wheel', 'python_version': 'cp36', 'requires_python': '>=4', 'size': 15599590, 'upload_time': '2021-01-05T17:19:38', 'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'yanked': False, 'yanked_reason': None}
    assert not reuse._is_version_satisfied(info, sys_version)

    # pure python
    info = {'comment_text': '', 'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 'downloads': -1, 'filename': 'google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'has_sig': False, 'md5_digest': '2651049b70d2ec07d8afd7637f198807', 'packagetype': 'bdist_wheel', 'python_version': 'source', 'requires_python': '>=3.6', 'size': 15599590, 'upload_time': '2021-01-05T17:19:38', 'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/google.protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'yanked': False, 'yanked_reason': None}
    assert reuse._is_version_satisfied(info, sys_version)

@pytest.mark.skipif(list(sys.version_info)[0:2] >= [3, 10],
    reason="no binary distribution of google.protobuf is available for python >= 3.10 on Windows")
def test_find_windows_artifact(reuse):
    package_name = "protobuf"
    target_version = "3.17.3"
    response = requests.get(f"https://pypi.org/pypi/{package_name}/{target_version}/json").json()
    assert reuse._find_matching_artifact(response["urls"])

def test_parse_filename(reuse):
    assert reuse._parse_filename("protobuf-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl") == {'distribution': 'protobuf', 'version': '1.19.5', 'build_tag': None, 'python_tag': 'cp36', 'abi_tag': 'cp36m', 'platform_tag': 'macosx_10_9_x86_64', 'ext': 'whl'}

def test_classic_import_no_version(reuse):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = reuse("mmh3", modes=reuse.fatal_exceptions)
        assert issubclass(w[-1].category, reuse.AmbiguityWarning)

def test_classic_import_same_version(reuse):
    version = reuse.Version(__import__("mmh3").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mod = reuse("mmh3", version=version, modes=reuse.fatal_exceptions)
        assert not w
        assert reuse.Version(mod.__version__) == version

def test_classic_import_diff_version(reuse):
    version = reuse.Version(__import__("mmh3").__version__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        major, minor, patch = version
        mod = reuse("mmh3", version=reuse.Version(major=major, minor=minor, patch=patch +1), modes=reuse.fatal_exceptions)
        assert issubclass(w[-1].category, reuse.VersionWarning)
        assert reuse.Version(mod.__version__) == version

def test_use_ugrade_version_warning(reuse):
    version = "0.0.0"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # no other way to change __version__ before the actual import while the version check happens on import
        test_use = reuse(reuse.Path("../src/use/use.py"), initial_globals={"test_version":version})
        assert test_use.test_version == test_use.__version__ == version
        assert w[0].category.__name__ == reuse.VersionWarning.__name__
