import builtins
import re
import sys
from collections import defaultdict, deque
from importlib.util import spec_from_loader
from inspect import getmembers, isclass
from logging import getLogger
from types import ModuleType
from typing import Any, Callable
from warnings import warn

from beartype import beartype

log = getLogger(__name__)

from use import AmbiguityWarning

packages_excluded_from_aspectizing: set = {}
"Set of packages that should be excluded from decoration."
modules_excluded_from_aspectizing: set = {}
"Set of modules that should be excluded from decoration."
_applied_decorators: dict[str, deque[Callable]] = defaultdict(deque)
"{qualname: [callable]} - to see which decorators are applied, in which order"
_aspectized_functions: dict[str, deque[Callable]] = defaultdict(deque)
"{qualname: [callable]} - the actually decorated functions to undo aspectizing"


def _apply_aspect(
    thing,
    check,
    pattern,
    decorator,
    aspectize_dunders=False,
    ignore_boundaries=False,
    recursive=True,
    force=False,
    level=0,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    for name, obj in getmembers(thing):
        if name in (
            "__init_subclass__",
            "__new__",
            "__prepare__",
        ):
            continue
        obj = getattr(obj, "__wrapped__", obj)
        # by using `[-2:-1][-1]` we avoid an IndexError if there
        # is only one item in the `__mro__` (e.g. `object`)
        mod = lookup_module(obj)
        if mod is None:
            continue
        module_name, loader, spec = get_module_info(mod)
        is_package, package_name = get_package(mod)
        ignore_boundaries |= int(level > 0)
        if level < 2 and isclass(obj):
            if module_name == "use.pypi_model":
                continue
            log.debug(
                f"recursing on {str(obj)[:20]} [{type(obj)}] with ({check}, {pattern} {decorator.__qualname__}"
            )
            _apply_aspect(
                obj,
                check=check,
                pattern=pattern,
                decorator=decorator,
                aspectize_dunders=aspectize_dunders,
                ignore_boundaries=ignore_boundaries,
                recursive=True,
                force=force,
                level=level + 1,
            )
            log.debug("finished recursion on %s", obj)
        # first let's exclude everything that doesn't belong to the module to avoid infinite recursion and other nasty stuff
        if mod is None and not ignore_boundaries:
            log.debug(
                f"skipped {obj} because {getattr(obj, '__module__', None)} and {thing} {getattr(thing, '__module__', None)} are not the same."
            )
            continue
        # we can't really tell where stuff is coming from if __package__ isn't set, so let's refuse the temptation to guess
        if package_name is None and not ignore_boundaries:
            raise AmbiguityWarning(
                f"""Package of {obj} and its parent {thing} are not defined, thus we can't reliably enforce module boundaries.
                If you want to fix this issue, make sure that obj.__package__ is set or if you want to ignore this issue,  
                use._apply_aspect with ignore_boundaries=True."""
            )
        # then there are things that we really shouldn't aspectize (up for the user to fill)
        if (
            module_name in modules_excluded_from_aspectizing
            or package_name in packages_excluded_from_aspectizing
        ):
            log.debug(
                f"{str(obj)[:20]} [{type(obj)}] is skipped because {obj.__module__} or {obj.__package__} are excluded. "
            )
            continue
        if not aspectize_dunders and name.startswith("__") and name.endswith("__"):
            log.debug(f"{str(obj)[:20]} [{type(obj)}] is skipped because it's a dunder")
            continue
        if check(obj) and re.match(pattern, name):
            if decorator in _applied_decorators[str(obj)[:20]] and not force:
                warn(
                    f"Applied decorator {decorator} to {str(obj)[:20]} [{type(obj)}] multiple times! If you intend this behaviour, use _apply_aspect with force=True"
                )
            _applied_decorators[obj.__qualname__].append(decorator)
            _aspectized_functions[obj.__qualname__] = obj
            try:
                thing.__dict__[name] = decorator(obj)
            except TypeError:
                continue
            log.info(
                f"{decorator.__qualname__} @ {obj.__module__}::{obj.__qualname__} [{obj.__class__.__name__}]"
            )
    return thing


def lookup_module(object):
    object_mod = None
    if isinstance(object, ModuleType):
        object_mod = builtins
    elif hasattr(object, "__module__"):
        if isinstance(object.__module__, str):
            object_mod = sys.modules[object.__module__]
        elif isinstance(object.__module__, ModuleType):
            object_mod = object.__module__
        else:
            return None
    elif hasattr(object, "__init__") and hasattr(object.__init__, "__module__"):
        object_mod = sys.modules[object.__init__.__module__]
    elif not isinstance(object, type):
        object_mod = builtins
    if not object_mod:
        raise NotImplementedError(
            "Don't know how to find module for '%s.%s' objects" % (object.__module__, object.__qualname__)
        )
    return object_mod


# @lru_cache(maxsize=0)
def get_module_info(mod: ModuleType):
    module_name = getattr(mod, "__name__", None) or mod.__spec__.name
    loader = getattr(mod, "__loader__", None) or mod.__spec__.loader
    spec = getattr(mod, "__spec__", None) or spec_from_loader(module_name, loader)
    return module_name, loader, spec


def get_package(mod):
    module_name, loader, spec = get_module_info(mod)
    module_parts = module_name.split(".")
    try:
        is_package = loader.is_package(module_name)
    except ImportError:
        is_package = False
    if is_package:
        package_name = module_name
    else:
        package_name = getattr(mod, "__package__", None) or ".".join(module_parts[:-1])
    return is_package, package_name


def isbeartypeable(thing):
    if type(thing).__name__.startswith("builtin_method"):
        return False
    if thing is type or thing is object or not hasattr(thing, "__dict__"):
        return False

    try:
        beartype(thing)
        return True
    except:
        return False
