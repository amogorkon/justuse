from packaging import tags
import requests

package_name = "numpy"
target_version = "1.19.5"
response = requests.get(f"https://pypi.org/pypi/{package_name}/{target_version}/json").json()
info = {'comment_text': '', 'digests': {'md5': '2651049b70d2ec07d8afd7637f198807', 'sha256': 'cc6bd4fd593cb261332568485e20a0712883cf631f6f5e8e86a52caa8b2b50ff'}, 'downloads': -1, 'filename': 'numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'has_sig': False, 'md5_digest': '2651049b70d2ec07d8afd7637f198807', 'packagetype': 'bdist_wheel', 'python_version': 'cp36', 'requires_python': '>=4', 'size': 15599590, 'upload_time': '2021-01-05T17:19:38', 'upload_time_iso_8601': '2021-01-05T17:19:38.152665Z', 'url': 'https://files.pythonhosted.org/packages/6a/9d/984f87a8d5b28b1d4afc042d8f436a76d6210fb582214f35a0ea1db3be66/numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl', 'yanked': False, 'yanked_reason': None}

from typing import Optional
import re

def parse_filename(filename:str) -> Optional[dict]:
    """Match the filename and return a dict of parts.
    >>> parse_filename(...)  # TODO add a proper doctest
    {"distribution": .., "version": .., ...} 
    """
    assert isinstance(filename, str)
    match:Optional[re.Match] = re.match(
        "(?P<distribution>.*)-"
        "(?P<version>.*)"
        "(?:-(?P<build_tag>.*))?-"
        "(?P<python_tag>.*)-"
        "(?P<abi_tag>.*)-"
        "(?P<platform_tag>.*)\\."
        "(?P<ext>whl|zip|tar|tar\\.gz)",
        filename
    )
    return match.groupdict() if match else None
