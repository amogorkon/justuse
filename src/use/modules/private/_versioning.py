"""
Module that will hold version specific functions

Author: ...
Date: 22.Oct.2021
"""

import sys
import platform
from packaging import tags
from functools import cache, reduce
from types import ModuleType
from typing import Callable, Union

from packaging.specifiers import SpecifierSet

from ._logging import log
from ..Decorators import pipes
from ..PlatformTag import PlatformTag
from ..get_supported import get_supported
from ...pypi_model import PyPI_Project, PyPI_Release, Version


def _sys_version():
    return Version(".".join(map(str, sys.version_info[0:3])))


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


# used inside _filter_by_platform
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


def _filter_by_version(project: PyPI_Project, version: str) -> PyPI_Project:
    return PyPI_Project(
        **{
            **project.dict(),
            **{
                "releases": {version: [v.dict() for v in project.releases[version]]}
                if project.releases.get(version)
                else {}
            },
        }
    )


@pipes
def _filtered_and_ordered_data(
    data: PyPI_Project, version: Version = None
) -> list[PyPI_Release]:
    if version:
        filtered = (
            data
            >> _filter_by_version(version)
            >> _filter_by_platform(tags=get_supported(), sys_version=_sys_version())
        )
    else:
        filtered = _filter_by_platform(
            data, tags=get_supported(), sys_version=_sys_version()
        )

    flat = reduce(list.__add__, filtered.releases.values(), [])
    return sorted(flat, key=lambda r: (not r.is_sdist, r.version), reverse=True)


def _filter_by_platform(
    project: PyPI_Project, tags: frozenset[PlatformTag], sys_version: Version
) -> PyPI_Project:
    filtered = {
        ver: [
            rel.dict()
            for rel in releases
            if _is_compatible(
                rel,
                sys_version=sys_version,
                platform_tags=tags,
                include_sdist=True,
            )
        ]
        for ver, releases in project.releases.items()
    }

    return PyPI_Project(**{**project.dict(), **{"releases": filtered}})


def _pebkac_no_version_hash(
    func=None, *, name: str, **kwargs
) -> Union[ModuleType, Exception]:

    if func:
        result = func(name=name, **kwargs)
        if isinstance(result, ModuleType):
            return result
    return RuntimeWarning(Message.cant_import_no_version(name))


def _pebkac_version_no_hash(
    func=None,
    *,
    name: str,
    version: Version,
    hash_algo,
    package_name: str,
    message_formatter: Callable[
        [str, str, Version, set[str]], str
    ] = Message.pebkac_missing_hash,
    **kwargs,
) -> Union[Exception, ModuleType]:
    if func:
        result = func(**all_kwargs(_pebkac_version_no_hash, locals()))
        if isinstance(result, ModuleType):
            return result
    try:
        hashes = {
            hash_alphabet.hexdigest_as_JACK(entry.digests.get(hash_algo.name))
            for entry in (_get_package_data(package_name).releases[version])
        }
        if not hashes:
            rw = RuntimeWarning(Message.pebkac_unsupported(package_name))
        else:
            rw = RuntimeWarning(message_formatter(name, package_name, version, hashes))
        rw.name = name
        rw.version = version
        rw.hashes = hashes
        return rw
    except (IndexError, KeyError) as ike:
        return RuntimeWarning(Message.no_distribution_found(package_name, version))


@pipes
def _pebkac_no_version_no_hash(*, name, package_name, hash_algo, **kwargs) -> Exception:
    # let's try to make an educated guess and give a useful suggestion
    data = _get_package_data(package_name) >> _filter_by_platform(
        tags=get_supported(), sys_version=_sys_version()
    )
    flat = reduce(list.__add__, data.releases.values(), [])
    priority = sorted(flat, key=lambda r: (not r.is_sdist, r.version), reverse=True)

    for info in priority:
        return _pebkac_version_no_hash(
            func=None,
            name=name,
            version=info.version,
            hash_algo=hash_algo,
            package_name=package_name,
            message_formatter=Message.no_version_or_hash_provided,
        )

    rw = RuntimeWarning(Message.pebkac_unsupported(package_name))
    rw.name = package_name
    return rw
