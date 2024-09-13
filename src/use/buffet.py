# noqa: E701
# here we're building the buffet of the future with pattern matching (>=3.10)

from logging import getLogger

from use.messages import UserMessage as Message
from use.pimp import (
    _auto_install,
    _ensure_version,
    _import_public_no_install,
    _pebkac_no_hash,
    _pebkac_no_version,
    _pebkac_no_version_no_hash,
)
from use.tools import pipes

log = getLogger(__name__)


# fmt: off
@pipes
def buffet_table(case, kwargs):
    match case:
#            +-------------------------- version specified?
#            |  +----------------------- hash specified?
#            |  |  +-------------------- publicly available?
#            |  |  |  +----------------- auto-install requested?
#            |  |  |  |
#            v  v  v  v
        case 1, 1, 1, 1: return _import_public_no_install(**kwargs) >> _ensure_version(**kwargs) >> _auto_install(**kwargs) # noqa: E701
        case 1, 1, 0, 1: return _auto_install(**kwargs) # noqa: E701
        case 1, _, 1, 0: return _import_public_no_install(**kwargs) >> _ensure_version(**kwargs) # noqa: E701
        case 0, 0, _, 1: return _pebkac_no_version_no_hash(**kwargs) # noqa: E701
        case 1, 0, _, 1: return _pebkac_no_hash(**kwargs) # noqa: E701
        case 0, 1, _, 1: return _pebkac_no_version(**kwargs) # noqa: E701
        case 0, _, 1, 0: return _import_public_no_install(**kwargs) # noqa: E701
        case _, _, 0, 0: return ImportError(Message.cant_import(**kwargs))  # noqa: E701
# fmt: on
