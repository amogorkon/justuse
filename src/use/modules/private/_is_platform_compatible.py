"""
Used in _is_compatible

Author: ...
Date: 21.Oct.2021
"""

import sys
import platform
from packaging import tags

from packaging.specifiers import SpecifierSet

from ..PlatformTag import PlatformTag
from ..Decorators import pipes
from ...pypi_model import PyPI_Release, Version


@pipes
def _is_platform_compatible(
    info: PyPI_Release, platform_tags: frozenset[PlatformTag], include_sdist=False
) -> bool:

    if "py2" in info.justuse.python_tag and "py3" not in info.justuse.python_tag:
        return False

    if not include_sdist and (
        ".tar" in info.justuse.ext or info.justuse.python_tag in ("cpsource", "sdist")
    ):
        return False

    if "win" in info.packagetype and sys.platform != "win32":
        return False

    if "win32" in info.justuse.platform_tag and sys.platform != "win32":
        return False

    if "macosx" in info.justuse.platform_tag and sys.platform != "darwin":
        return False

    our_python_tag = tags.interpreter_name() + tags.interpreter_version()
    supported_tags = set(
        [
            our_python_tag,
            "py3",
            "cp3",
            f"cp{tags.interpreter_version()}",
            f"py{tags.interpreter_version()}",
        ]
    )

    given_platform_tags = (
        info.justuse.platform_tag.split(".") << map(PlatformTag) >> frozenset
    )

    if info.is_sdist and info.requires_python is not None:
        given_python_tag = {
            our_python_tag
            for p in info.requires_python.split(",")
            if Version(platform.python_version()) in SpecifierSet(p)
        }
    else:
        given_python_tag = set(info.justuse.python_tag.split("."))

    # print(supported_tags, given_python_tag)

    return any(supported_tags.intersection(given_python_tag)) and (
        (info.is_sdist and include_sdist)
        or any(given_platform_tags.intersection(platform_tags))
    )
