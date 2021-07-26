import os
import sys
import warnings
from pathlib import Path
from setuptools import _find_all_simple
from unittest import skip

import pytest
import requests
from mypy.__main__ import console_entry
from yarl import URL

from .unit_test import reuse

not_local = not Path("local_switch").exists()


# Add in-progress tests here


def test_template(reuse):
    pass


def test_is_platform_compatible_macos(reuse):
    platform_tag = list(map(lambda i: i.platform, reuse.use.get_supported()))[0]
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
    platform_tags = set(map(lambda i: i.platform, reuse.use.get_supported()))
    assert reuse._is_platform_compatible(info, platform_tags)


def test_is_platform_compatible_win(reuse):
    platform_tag = list(map(lambda i: i.platform, reuse.use.get_supported()))[0]
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
    platform_tags = set(map(lambda i: i.platform, reuse.use.get_supported()))
    assert reuse._is_platform_compatible(info, platform_tags)

@pytest.mark.xfail(not not_local, reason="Incomplete type hints")
def test_types():
    files = list(filter(
      lambda p: p.endswith(".py"),
      _find_all_simple("./src")
    ))
    exit_code:int = None
    prev_exit:Callable[int, ...] = sys.exit
    sys.exit:Callable[int, ...] = lambda *args: \
        exec("global exit_code; exit_code=args[0]")
    try:
        prev_argv = sys.argv
        sys.argv = ["-m", *files]
        console_entry()
        assert exit_code == 0, "mypy completed with error(s)"
    finally:
      sys.argv = prev_argv
      sys.exit = prev_exit
