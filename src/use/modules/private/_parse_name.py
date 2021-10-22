import re
from logging import getLogger

log = getLogger(__name__)


def _parse_name(name) -> tuple[str, str]:
    ret = name, name
    try:
        match = re.match(r"(?P<package_name>[^/.]+)/?(?P<rest>[a-zA-Z0-9._]+)?$", name)
        assert match, f"Invalid name spec: {name!r}"
        names = match.groupdict()
        package_name = names["package_name"]
        rest = names["rest"]
        if not package_name:
            package_name = rest
        if not rest:
            rest = package_name
        ret = (package_name, rest)
        return ret
    finally:
        log.info("_parse_name(%s) -> %s", repr(name), repr(ret))
