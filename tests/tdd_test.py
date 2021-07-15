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
def test_parse_filename(reuse):
    assert reuse.parse_filename("numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl") == ("numpy", "1.19.5", "cp36", "cp36m", "macosx_10_9_x86_64")

@pytest.mark.xfail(not_local, reason="in development") 
def test_is_platform_compatible(reuse):
    info = {'comment_text': '', 'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 'downloads': -1, 'filename': 'numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'has_sig': False, 'md5_digest': '2651049b70d2ec07d8afd7637f198807', 'packagetype': 'bdist_wheel', 'python_version': 'source', 'requires_python': '>=3.6', 'size': 15599590, 'upload_time': '2021-01-05T17:19:38', 'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'yanked': False, 'yanked_reason': None}
    reuse.parse
    assert reuse.is_platform_compatible(info, sys.platform)
