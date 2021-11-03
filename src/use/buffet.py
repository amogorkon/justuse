# here we're building the buffet of the future with pattern matching (>=3.10)

from logging import getLogger

log = getLogger(__name__)

from use import pimp
from use.messages import Message

# fmt: off

def buffet_table(case, kwargs):
    name = kwargs.get('name')
    case_func = {
        (0, 0, 0, 0): lambda: ImportError(Message.cant_import(name)),
        (0, 0, 0, 1): lambda: pimp._pebkac_no_version_no_hash(**kwargs),
        (0, 0, 1, 0): lambda: pimp._import_public_no_install(**kwargs),
        (0, 1, 0, 0): lambda: ImportError(Message.cant_import(name)),
        (1, 0, 0, 0): lambda: ImportError(Message.cant_import(name)),
        (0, 0, 1, 1): lambda: pimp._auto_install(
            func=lambda: pimp._import_public_no_install(**kwargs), **kwargs
        ),
        (0, 1, 1, 0): lambda: pimp._import_public_no_install(**kwargs),
        (1, 1, 0, 0): lambda: ImportError(Message.cant_import(name)),
        (1, 0, 0, 1): lambda: pimp._pebkac_version_no_hash(**kwargs),
        (1, 0, 1, 0): lambda: pimp._ensure_version(pimp._import_public_no_install(**kwargs)),
        (0, 1, 0, 1): lambda: pimp._pebkac_no_version_hash(**kwargs),
        (0, 1, 1, 1): lambda: pimp._pebkac_no_version_hash(pimp._import_public_no_install, **kwargs),
        (1, 0, 1, 1): lambda: pimp._ensure_version(
            pimp._pebkac_version_no_hash(func=lambda: pimp._import_public_no_install(**kwargs), **kwargs),
        ),
        (1, 1, 0, 1): lambda: pimp._auto_install(**kwargs),
        (1, 1, 1, 0): lambda: pimp._ensure_version(pimp._import_public_no_install(**kwargs)),
        (1, 1, 1, 1): lambda: pimp._auto_install(
            func=lambda: pimp._ensure_version(pimp._import_public_no_install(**kwargs)), **kwargs
        ),
    }[case]
    log.info("case_func = '%s' %s", case_func.__qualname__, case_func)
    log.info("kwargs = %s", repr(kwargs))
    result = case_func()
    log.info("result = %s", repr(result))
# fmt: on
