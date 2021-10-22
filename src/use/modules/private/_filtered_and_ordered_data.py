from functools import reduce

from ._filter_by_version import _filter_by_version
from ._filter_by_platform import _filter_by_platform
from ._sys_version import _sys_version
from ..get_supported import get_supported
from ..Decorators import pipes
from ...pypi_model import PyPI_Project, PyPI_Release, Version


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
