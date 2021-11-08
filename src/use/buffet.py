# here we're building the buffet of the future with pattern matching (>=3.10)

from logging import getLogger

log = getLogger(__name__)

from use.messages import Message
from use.pimp import (_auto_install, _ensure_version,
                      _import_public_no_install, _pebkac_no_hash,
                      _pebkac_no_version, _pebkac_no_version_no_hash)


def buffet_table(case, kwargs):
    match case: # type: ignore
        case _, _, 0, 0: return ImportError(Message.cant_import(**kwargs))
        case 0, _, 1, 0: return _import_public_no_install(**kwargs)
        case 1, _, 1, 0: return _import_public_no_install(**kwargs) >> _ensure_version(**kwargs)
        case 0, 0, _, 1: return _pebkac_no_version_no_hash(**kwargs)
        case 1, 0, _, 1: return _pebkac_no_hash(**kwargs)
        case 0, 1, _, 1: return _pebkac_no_version(**kwargs)
        case 1, 1, 0, 1: return _auto_install(**kwargs)
        case 1, 1, 1, 1: return _import_public_no_install(**kwargs) >> _ensure_version(**kwargs) >> _auto_install(**kwargs)
