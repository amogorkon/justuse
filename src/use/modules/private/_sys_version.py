import sys
from ...pypi_model import Version


def _sys_version():
    return Version(".".join(map(str, sys.version_info[0:3])))
