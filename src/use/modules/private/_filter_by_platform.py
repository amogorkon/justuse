from ._is_compatible import _is_compatible
from ..PlatformTag import PlatformTag

from ...pypi_model import PyPI_Project, Version


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
