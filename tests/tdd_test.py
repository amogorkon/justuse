import os
import sys
import warnings
from pathlib import Path
from unittest import skip

import pytest
import requests
from packaging.version import Version
from yarl import URL

from unit_test import reuse

not_local = not Path("local_switch").exists()


# Add in-progress tests here

@pytest.mark.xfail(not_local, reason="in development") 
def test_template(reuse):
    pass

@pytest.mark.xfail(not_local, reason="in development") 
def test_is_platform_compatible_macos(reuse):
    info = {'comment_text': '', 
            'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 
            'downloads': -1, 
            'filename': 'numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 
            'has_sig': False, 
            'md5_digest': '2651049b70d2ec07d8afd7637f198807', 
            'packagetype': 'bdist_wheel', 
            'python_version': 'source', 
            'requires_python': '>=3.6', 
            'size': 15599590, 
            'upload_time': '2021-01-05T17:19:38', 
            'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 
            'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 
            'yanked': False, 
            'yanked_reason': None}
    platform_tags = {'win_amd64',}
    assert reuse._is_platform_compatible(info, platform_tags)

@skip
@pytest.mark.xfail(not_local, reason="in development") 
def test_is_platform_compatible_win(reuse):
    info = {'comment_text': '',
        'digests': {'md5': 'baf1bd7e3a8c19367103483d1fd61cfc', 'sha256': 'dbd18bcf4889b720ba13a27ec2f2aac1981bd41203b3a3b27ba7a33f88ae4827'},
        'downloads': -1,
        'filename': 'numpy-1.19.5-cp36-cp36m-win_amd64.whl',
        'has_sig': False,
        'md5_digest': 'baf1bd7e3a8c19367103483d1fd61cfc',
        'packagetype': 'bdist_wheel',
        'python_version': 'cp36',
        'requires_python': '>=3.6',
        'size': 13227547,
        'upload_time': '2021-01-05T17:24:53',
        'upload_time_iso_8601': '2021-01-05T17:24:53.052845Z',
        'url': 'https://files.pythonhosted.org/packages/ea/bc/da526221bc111857c7ef39c3af670bbcf5e69c247b0d22e51986f6d0c5c2/numpy-1.19.5-cp36-cp36m-win_amd64.whl',
        'yanked': False,
        'yanked_reason': None}
    platform_tags = {'win_amd64',}
    assert reuse._is_platform_compatible(info, platform_tags)

@pytest.mark.xfail(not_local, reason="in development")
def test_classic_import_no_version(reuse):
 with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mod = reuse("mmh3", fatal_exceptions=True)
    assert issubclass(w[-1].category, reuse.AmbiguityWarning)

@pytest.mark.xfail(not_local, reason="in development")
def test_classic_import_same_version(reuse):
 version = Version(__import__("mmh3").__version__)
 with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mod = reuse("mmh3", version=ver, fatal_exceptions=True)
    assert not w
    assert mod.__version__ == cur_ver_tup

@pytest.mark.xfail(not_local, reason="in development")
def test_classic_import_diff_version(reuse):
 version = Version(__import__("mmh3").__version__)
 with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    mod = reuse("mmh3", version=str(version), fatal_exceptions=True)
    assert issubclass(w[-1].category, reuse.VersionWarning)
    assert Version(mod.__version__) == version
