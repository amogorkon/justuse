"""
Used in _is_compatible

Author: ...
Date: 21.Oct.2021
"""


from functools import cache
from logging import getLogger
from packaging.specifiers import SpecifierSet


log = getLogger(__name__)


@cache
def _is_version_satisfied(specifier: str, sys_version) -> bool:
    """
    SpecifierSet("") matches anything, no need to artificially
    lock down versions at this point

    @see https://warehouse.readthedocs.io/api-reference/json.html
    @see https://packaging.pypa.io/en/latest/specifiers.html
    """
    specifiers = SpecifierSet(specifier or "")
    is_match = sys_version in specifiers

    log.debug("is_version_satisfied(info=i)%s in %s", sys_version, specifiers)
    return is_match
