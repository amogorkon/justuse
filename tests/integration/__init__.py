import os
import sys

from justuse import Path
from justuse import use as use

src = import_base = Path(__file__).parent / "../../src"
cwd = Path().cwd()
os.chdir(src)
sys.path.insert(0, "") if "" not in sys.path else None

if sys.version_info < (3, 9) and "use" not in sys.modules:
    import gc
    import types
    import typing
    from abc import ABCMeta
    from collections.abc import Callable
    from types import CellType
    from typing import _GenericAlias as GenericAlias

    for t in (list, dict, set, tuple, frozenset, ABCMeta, Callable, CellType):
        r = gc.get_referents(t.__dict__)[0]
        r.update({
            "__class_getitem__": classmethod(GenericAlias),
        })


os.chdir(cwd)
