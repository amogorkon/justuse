"""
Used in _auto_install

Author: ...
Date: 21.Oct.2021
"""


import sys
import zipimport
from importlib.machinery import SourceFileLoader


def _clean_sys_modules(package_name: str) -> None:
    for k in dict(
        [
            (k, _ensure_loader(v))
            for k, v in sys.modules.items()
            if (
                getattr(v, "__spec__", None) is None
                or isinstance(
                    _ensure_loader(v), (SourceFileLoader, zipimport.zipimporter)
                )
            )
            and package_name in k.split(".")
        ]
    ):
        if k in sys.modules:
            del sys.modules[k]
