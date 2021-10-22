import packaging
from functools import cache

from logging import getLogger
from pip._internal.utils import compatibility_tags

from .PlatformTag import PlatformTag

log = getLogger(__name__)


@cache
def get_supported() -> frozenset[PlatformTag]:
    log.debug("enter get_supported()")
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, package_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    items: list[PlatformTag] = []

    for tag in compatibility_tags.get_supported():
        items.append(PlatformTag(platform=tag.platform))
    for tag in packaging.tags._platform_tags():
        items.append(PlatformTag(platform=str(tag)))

    tags = frozenset(items)
    log.debug("leave get_supported() -> %s", repr(tags))
    return tags
