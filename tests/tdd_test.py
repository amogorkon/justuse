import importlib
import os
import re
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from unittest import skip

import pytest
import requests
from setuptools import _find_all_simple

from .unit_test import log, reuse

not_local = "GITHUB_REF" in os.environ


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
        "url": "https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp3{sys.version_info[1]}-cp3{sys.version_info[1]}m-{platform_tag}.whl",
        "yanked": False,
        "yanked_reason": None,
    }
    assert reuse._is_platform_compatible(info, platform_tags)


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
    }
    assert reuse._is_platform_compatible(info, platform_tags, include_sdist=False)



def _do_load_venv_mod(reuse, package, version=None):
    if not version:
        data = reuse._get_filtered_data(reuse._get_package_data(package)) 
        versions = list(data["releases"].keys())
        version = versions[-1]
    mod = reuse._load_venv_mod(package, version)
    log.info("_load_venv_mod(%r, %r): %s", package, version, mod)
    assert mod
    assert mod.__version__ == version


@pytest.mark.skipif(sys.platform.startswith("win"), reason="windows venv package metadata")
def test_load_venv_mod_protobuf(reuse):
    _do_load_venv_mod(reuse, "protobuf")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="windows venv package metadata")
def test_load_venv_mod_numpy(reuse):
    _do_load_venv_mod(reuse, "numpy", "1.19.3")


def test_db_setup(reuse):
    assert reuse.registry


@pytest.mark.skipif(True, reason="in development")
def test_unsupported_artifact(reuse):
    hashes = {
        "win": "1fdae7d980a2fa617d119d0dc13ecb5c23cc63a8b04ffcb5298f2c59d86851e9",
        "linux": "36a089dc604032d41343d86290ce85d4e6886012eea73faa88001260abf5ff81",
        "macos": "39b5d36ab71f73c068cdcf70c38075511de73616e6c7fdd112d6268c2704d9f5",
    }
    if sys.platform.startswith("win"):
        del hashes["win"]
    elif sys.platform.startswith("macos"):
        del hashes["macos"]
    else:
        del hashes["linux"]
    np = reuse(
        "sqlalchemy",
        version="1.4.22",
        hashes="5de64950137f3a50b76ce93556db392e8f1f954c2d8207f78a92d1f79aa9f737",
        modes=reuse.auto_install,
    )
    assert False, np.__version__