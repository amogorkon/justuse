import codecs
import linecache
from types import ModuleType
from typing import Any, Optional

from . import _parse_name


def _build_mod(
    *,
    name,  # TODO: this should be a package name and module name
    code,
    initial_globals: Optional[dict[str, Any]],
    module_path,
) -> ModuleType:

    package_name, rest = _parse_name(name)
    mod = ModuleType(rest)

    mod.__dict__.update(initial_globals or {})
    mod.__file__ = str(module_path)
    mod.__path__ = [str(module_path.parent)]
    mod.__package__ = package_name
    mod.__name__ = rest
    code_text = codecs.decode(code)
    # module file "<", ">" chars are specially handled by inspect
    getattr(linecache, "cache")[module_path] = (
        len(code),  # size of source code
        None,  # last modified time; None means there is no physical file
        [
            *map(lambda ln: ln + "\x0a", code_text.splitlines())
        ],  # a list of lines, including trailing newline on each
        mod.__file__,  # file name, e.g. "<mymodule>" or the actual path to the file
    )
    # not catching this causes the most irritating bugs ever!
    try:
        codeobj = compile(code, module_path, "exec")
        exec(codeobj, mod.__dict__)
    except:  # reraise anything without handling - clean and simple.
        raise
    return mod
