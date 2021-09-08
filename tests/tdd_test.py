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


def test_pure_python_package(reuse):
    # https://pypi.org/project/example-pypi-package/
    file = (
        reuse.Path.home()
        / ".justuse-python/packages/example_pypi_package-0.1.0-py3-none-any.whl"
    )

    file.unlink(missing_ok=True)
    test = reuse(
        "example-pypi-package.examplepy",
        version="0.1.0",
        hashes={
            "3c1b4ddf718d85bde796a20cf3fdea254a33a4dc89129dff5bfc5b7cd760c86b",
            "ce89b1fe92abc55b4349bc58462ba255c42132598df6fe3a416a75b39b872a77",
        },
        modes=reuse.auto_install,
    )
    assert str(test.Number(2)) == "2"
    if file.exists():
        file.unlink()


def _do_load_venv_mod(reuse, name):
    data = reuse._get_filtered_data(reuse._get_package_data(name))
    versions = sorted(list(data["releases"].keys()))
    version = versions[-1]
    items = data["releases"][version]
    mod = None
    for item in items:
            mod = reuse._load_venv_mod(
                name=name,
                version=item["version"],
            )
            if mod:
                return
    assert False


def test_load_venv_mod_protobuf(reuse):
    _do_load_venv_mod(reuse, "protobuf")


def test_load_venv_mod_numpy(reuse):
    _do_load_venv_mod(reuse, "numpy")


def test_db_setup(reuse):
    assert reuse.registry


def _get_test_ver_hash_data(reuse):
    VerHash = reuse.VerHash
    h = "5de64950137f3a50b76ce93556db392e8f1f954c2d8207f78a92d1f79aa9f737"
    vh1, vh2 = (VerHash("1.0.1", h), VerHash("1.0.2", h))
    vh1u, vh2u, vh3u = (VerHash("1.0.1", None), VerHash(None, h), VerHash(None, None))
    vh1b = VerHash("1.0.1", h)
    return (VerHash, h, vh1, vh2, vh1u, vh2u, vh3u, vh1b)


def test_ver_hash_1(reuse):
    VerHash, h, vh1, vh2, vh1u, vh2u, vh3u, vh1b = _get_test_ver_hash_data(reuse)
    assert vh1 and vh2
    assert vh1.hash == vh2.hash
    assert vh1u.version
    assert vh2u.hash
    assert not vh3u
    assert vh1 == vh1b
    assert vh1 == vh1
    assert vh1 != ("1.0.1", None)
    assert vh1 != ("1.0.1", None, None)


def test_ver_hash_2(reuse):
    VerHash, h, vh1, vh2, vh1u, vh2u, vh3u, vh1b = _get_test_ver_hash_data(reuse)
    assert vh1 == ("1.0.1", h)
    assert vh1 != ("1.0.1", h, None)
    assert vh1 != object()
    assert ("1.0.1", h, None) != vh1
    assert "1.0.1" in vh1
    assert h in vh1
    assert "1.0.1" not in vh2
    assert h in vh2
    assert h in vh2u
    assert h not in vh3u