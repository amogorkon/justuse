"""
Module used inside _filter_by_platform

Author: ...
Date: 21.Oct.2021
"""


from ._is_version_satisfied import _is_version_satisfied
from ._is_platform_compatible import _is_platform_compatible
from ...pypi_model import PyPI_Release


def _is_compatible(
    info: PyPI_Release, sys_version, platform_tags, include_sdist=None
) -> bool:
    """Return true if the artifact described by 'info'
    is compatible with the current or specified system."""
    specifier = info.requires_python

    return (
        (not specifier or _is_version_satisfied(specifier, sys_version))
        and _is_platform_compatible(info, platform_tags, include_sdist)
        and not info.yanked
        and (include_sdist or info.justuse.ext not in ("tar", "tar.gz" "zip"))
    )
