"""
Shared utilities to break circular imports between use.pimp and use.classes.
"""

import codecs
import linecache
from importlib.machinery import ModuleSpec, SourceFileLoader
from types import ModuleType
from typing import Any


def _build_mod(
    *,
    mod_name,
    code: bytes,
    initial_globals: dict[str, Any] | None,
    module_path,
    pkg_name="",
) -> ModuleType:
    mod = ModuleType(mod_name)
    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__path__ = [str(module_path.parent)]
    mod.__package__ = pkg_name
    mod.__name__ = mod_name
    loader = SourceFileLoader(mod_name, str(module_path))
    mod.__loader__ = loader
    mod.__spec__ = ModuleSpec(mod_name, loader)
    code_text = codecs.decode(code)
    getattr(linecache, "cache")[module_path] = (
        len(code),
        None,
        [*map(lambda ln: ln + "\x0a", code_text.splitlines())],
        mod.__file__,
    )
    try:
        codeobj = compile(code, module_path, "exec")
        exec(codeobj, mod.__dict__)
    except:
        raise
    return mod


def _modules_are_compatible(pre, post):
    from justuse.pimp import _is_compatible

    for name, pre_obj in pre.__dict__.items():
        if callable(pre_obj):
            try:
                post_obj = getattr(post, name)
            except AttributeError:
                return False
            if _is_compatible(pre_obj, post_obj):
                continue
            else:
                return False
    return True
