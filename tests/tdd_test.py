import importlib
import os
import re
import shlex
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, distribution
from importlib.util import find_spec
from pathlib import Path
from shutil import rmtree
from unittest import skip

import pytest
import requests

from .unit_test import PyPI_Release, Version, log, reuse

not_local = "GITHUB_REF" in os.environ
is_win = sys.platform.lower().startswith("win")
not_win = not is_win

# Add in-progress tests here


def test_template(reuse):
    pass


def test_is_platform_compatible_macos(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "2651049b70d2ec07d8afd7637f198807",
            "sha256": "cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "2651049b70d2ec07d8afd7637f198807",
        "packagetype": "bdist_wheel",
        "python_version": "source",
        "requires_python": ">=3.6",
        "size": 15599590,
        "upload_time": "2021-01-05T17:19:38",
        "upload_time_iso_8601": "2021-01-05T17:19:38.152665Z",
        "url": f"https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
        "version": "1.19.5",
    }
    assert reuse._is_platform_compatible(PyPI_Release(**info), platform_tags)


def test_is_platform_compatible_win(reuse):
    platform_tags = reuse.use.get_supported()
    platform_tag = next(iter(platform_tags))
    info = {
        "comment_text": "",
        "digests": {
            "md5": "baf1bd7e3a8c19367103483d1fd61cfc",
            "sha256": "dbd18bcf4889b720ba13a27ec2f2aac1981bd41203b3a3b27ba7a33f88ae4827",
        },
        "downloads": -1,
        "filename": f"numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "has_sig": False,
        "md5_digest": "baf1bd7e3a8c19367103483d1fd61cfc",
        "packagetype": "bdist_wheel",
        "python_version": f"cp3{sys.version_info[1]}",
        "requires_python": f">=3.{sys.version_info[1]}",
        "size": 13227547,
        "upload_time": "2021-01-05T17:24:53",
        "upload_time_iso_8601": "2021-01-05T17:24:53.052845Z",
        "url": f"https://files.pythonhosted.org/packages/ea/bc/da526221bc111857c7ef39c3af670bbcf5e69c247b0d22e51986f6d0c5c2/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
        "version": "1.19.5",
    }
    assert reuse._is_platform_compatible(
        PyPI_Release(**info), platform_tags, include_sdist=False
    )


def test_pure_python_package(reuse):
    # https://pypi.org/project/example-pypi-package/
    file = (
        reuse.Path.home()
        / ".justuse-python/packages/example_pypi_package-0.1.0-py3-none-any.whl"
    )
    venv_dir = reuse.Path.home() / ".justuse-python/venv/example-pypi-package/0.1.0"
    file.unlink(missing_ok=True)
    if venv_dir.exists():
        rmtree(venv_dir)

    test = reuse(
        "example-pypi-package/examplepy",
        version="0.1.0",
        hashes={
            "3c1b4ddf718d85bde796a20cf3fdea254a33a4dc89129dff5bfc5b7cd760c86b",
            "ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77",
        },
        modes=reuse.auto_install,
    )
    assert (
        venv_dir.exists() == False
    ), "Should not have created venv for example-pypi-package"

    assert str(test.Number(2)) == "2"
    if file.exists():
        file.unlink()


def test_db_setup(reuse):
    assert reuse.registry


@pytest.mark.skipif(True, reason="broken")
def test_no_isolation(reuse):
    assert test_load_multi_version(reuse, "numpy", "1.19.0", 1)
    assert test_load_multi_version(reuse, "numpy", "1.19.0", 1)


def installed_or_skip(reuse, name, version=None):
    if not (spec := find_spec(name)):
        pytest.skip(f"{name} not installed")
        return False
    try:
        dist = distribution(spec.name)
    except PackageNotFoundError as pnfe:
        pytest.skip(f"{name} partially installed: {spec=}, {pnfe}")

    if not (
        (ver := dist.metadata["version"])
        and (not version or reuse.Version(version)) == (not ver or reuse.Version(ver))
    ):
        pytest.skip(f"found '{name}' v{ver}, but require v{version}")
        return False
    return True


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_str(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse("matplotlib/matplotlib.pyplot")
    assert mod


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_tuple(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse(("matplotlib", "matplotlib.pyplot"))
    assert mod


@pytest.mark.skipif(not_local, reason="requires matplotlib")
def test_use_kwargs(reuse):
    if not installed_or_skip(reuse, "matplotlib"):
        return
    mod = reuse(package_name="matplotlib", module_name="matplotlib.pyplot")
    assert mod
