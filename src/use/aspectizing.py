import builtins
import re
import sys
from collections import defaultdict, deque
from functools import wraps
from importlib.util import spec_from_loader
from inspect import getmembers, isclass
from logging import getLogger
from types import ModuleType
from typing import Any, Callable
from warnings import catch_warnings, filterwarnings, warn

from beartype import beartype
from beartype.roar import BeartypeCallHintPepParamException, BeartypeDecorHintPep585DeprecationWarning
from icontract import ensure, invariant, require

log = getLogger(__name__)

from use import AmbiguityWarning
from use.messages import _web_aspectizing_overview

_applied_decorators: dict[str, deque[Callable]] = defaultdict(deque)
"{qualname: [callable]} - to see which decorators are applied, in which order"
_aspectized_functions: dict[str, deque[Callable]] = defaultdict(deque)
"{qualname: [callable]} - the actually decorated functions to undo aspectizing"
def Dynamic_aspect(
    thing,
    check,
    pattern,
    qual_name,
    decorator,
    /,
    *,
    regex=None,
    aspectize_dunders=False,
    level=0,
    dry_run=False,
    visited: set[int] = None,
    excluded_names: set[str] = None,
    excluded_types: set[type] = None,
    hits: list[str] = None,
    last=True,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    if visited is None:
        visited = set()
    if id(thing) in visited:
        return
    if hits is None:
        hits = []
    if excluded_names is None:
        excluded_names = set()
    if excluded_types is None:
        excluded_types = set()
    if not regex:
        regex = re.compile(pattern, re.DOTALL)
    lvl = "".join([" "] * level)
    for name in object.__dir__(thing):
        obj = getattr(thing, name, None)
        if obj is None:
            continue
        name = getattr(obj, "__qualname__", None) or name
        if id(obj) in visited:
            continue
        visited.add(id(obj))
        if name in excluded_names or type(obj) in excluded_types or not check(obj) or not regex.match(name):
            continue
        module_name, loader, spec = get_module_info(lookup_module(obj))
        if dry_run:
            hits.append(f"{obj.__module__}::{getattr(obj, '__qualname__', name)}")
        else:
            try:
                wrapped = _apply_decorator(
                    thing=thing,
                    obj=obj,
                    decorator=decorator,
                    name=name,
                    module_name=module_name,
                )
                # Mustn't forget to track!
                visited.add(id(wrapped))
            except AttributeError:
                pass
            Dynamic_aspect(
                obj,
                check,
                pattern,
                decorator,
                regex=regex,
                aspectize_dunders=aspectize_dunders,
                excluded_names=excluded_names,
                excluded_types=excluded_types,
                visited=visited,
                level=level + 1,
                dry_run=dry_run,
                hits=hits,
                last=False,
            )

def aspect(
    thing,
    check,
    pattern,
    qual_name,
    decorator,
    /,
    *,
    regex=None,
    aspectize_dunders=False,
    recursive=True,
    level=0,
    dry_run=False,
    visited: set[int] = None,
    excluded_names: set[str] = None,
    excluded_types: set[type] = None,
    hits: list[str] = None,
    last=True,  # and first
) -> Any:
    __name__ = func.__module__.__name__
    qual_name = (func.__name__)
    """Apply the aspect as a side-effect, no copy is created."""
    if visited is None:
        visited = set()
    if id(thing) in visited:
        return

    if hits is None:
        hits = []

    if excluded_names is None:
        excluded_names = set()
    if excluded_types is None:
        excluded_types = set()

    # We compile once so in subsequent recursive calls, we have it already.
    if not regex:
        regex = re.compile(pattern, re.DOTALL)

    lvl = "".join([" "] * level)

    # object.__dir__ is the *only* reliable way to get all attributes
    # of an object that nobody can override,
    # dir() and inspect.getmembers both build on __dir__, but are not
    # reliable.
    for name in object.__dir__(thing):
        # object.__getattribute__ is asymmetric to object.__dir__,
        # skipping looking up things on the parent type
        obj = getattr(thing, name, None)
        # object.__dir__ returns *EVERYTHING*, including stuff that is only
        # there on demand but not really used, so we ignore those.

        if obj is None:
            continue

        # We can't really filter by name if it doesn't have a qualname
        name = getattr(obj, "__qualname__", None) or name

        if id(obj) in visited:
            continue

        # Time to get serious!
        visited.add(id(obj))

        # check main::ProxyModule.__matmul__ for defaults
        if name in excluded_names or type(obj) in excluded_types or not check(obj) or not regex.match(name):
            continue

        module_name, loader, spec = get_module_info(lookup_module(obj))

        if dry_run:
            hits.append(f"{obj.__module__}::{getattr(obj, '__qualname__', name)}")
        else:
            try:
                wrapped = _apply_decorator(
                    thing=thing,
                    obj=obj,
                    decorator=decorator,
                    name=name,
                    module_name=module_name,
                )
                # Mustn't forget to track!
                visited.add(id(wrapped))

            # AttributeError: readonly attribute
            except AttributeError:
                pass
        if recursive:
            log.debug(
                f"{lvl}recursing on {str(obj)[:20]} [{type(obj)}] "
                f"{lvl}with ({check}, {pattern} {decorator.__qualname__}"
            )

            aspect(
                obj,
                check,
                pattern,
                decorator,
                regex=regex,
                aspectize_dunders=aspectize_dunders,
                excluded_names=excluded_names,
                excluded_types=excluded_types,
                recursive=True,
                visited=visited,
                level=level + 1,
                dry_run=dry_run,
                hits=hits,
                last=False,
            )
            log.debug("finished recursion on %s", obj)

    # this the last thing in the original call, after all the recursion
    if last and dry_run:
        print("check your browser!")
        _web_aspectizing_overview(
            decorator=decorator,
            pattern=pattern,
            check=check,
            visited=visited,
            hits=hits,
        )
    return thing


def _apply_decorator(*, thing, obj, decorator, name, module_name):
    previous_object_id = id(obj)
    wrapped = decorator(obj)
    new_object_id = id(wrapped)

    _applied_decorators[new_object_id].extend(_applied_decorators[previous_object_id])
    _applied_decorators[new_object_id].append(decorator)

    _aspectized_functions[new_object_id].extend(_aspectized_functions[previous_object_id])
    _aspectized_functions[new_object_id].append(obj)

    setattr(thing, name, decorator(obj))

    # cleanup
    del _applied_decorators[previous_object_id]
    del _aspectized_functions[previous_object_id]
    log.info(f"Applied {decorator.__qualname__} to {module_name}::{name} [{obj.__class__.__qualname__}]")
    return wrapped


@require(lambda obj: obj is not None)
def lookup_module(obj) -> ModuleType:
    object_mod = None
    if isinstance(obj, ModuleType):
        object_mod = builtins
    elif hasattr(obj, "__module__"):
        if isinstance(obj.__module__, str):
            object_mod = sys.modules[obj.__module__]
        elif isinstance(obj.__module__, ModuleType):
            object_mod = obj.__module__
        else:
            return None
    elif hasattr(obj, "__init__") and hasattr(obj.__init__, "__module__"):
        object_mod = sys.modules[obj.__init__.__module__]
    elif not isinstance(obj, type):
        object_mod = builtins
    if not object_mod:
        raise NotImplementedError(
            "Don't know how to find module for '%s.%s' objects" % (obj.__module__, obj.__qualname__)
        )
    return object_mod


def get_module_info(mod: ModuleType):
    if mod is None:
        return None, None, None
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
    with catch_warnings():
        filterwarnings(action="ignore", category=BeartypeDecorHintPep585DeprecationWarning)
        if type(thing).__name__.startswith("builtin_method"):
            return False
        if thing in (type, object):
            return False
        try:
            beartype(thing)
            return True
        except:
            return False


def any_callable(thing):
    try:
        object.__getattribute__(thing, "__call__")
        return True
    except AttributeError:
        return False


def woody_logger(func: callable) -> callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """

    # @wraps(func)
    def wrapper(*args, **kwargs):
        # log.debug(f"{func.__name__}({args}, {kwargs})")
        # print(f"{args} {kwargs} -> {func.__module__}::{func.__qualname__}")
        res = func(*args, **kwargs)
        # print(type(res), res)
        # print(f"{func.__module__}::")
        # print(f"{func.__qualname__} -> ")
        # try:
        #     print(f'{res}')
        # except AttributeError:
        #     pass
        return res

    return wrapper


def deaspect(thing, name):
    obj = getattr(thing, name)
    unwrapped = _aspectized_functions[id(obj)].pop()
    _applied_decorators[id(obj)].pop()
    setattr(thing, name, unwrapped)

    _aspectized_functions[id(unwrapped)].extend(_aspectized_functions[id(obj)])
    _applied_decorators[id(unwrapped)].extend(_applied_decorators[id(obj)])

    del _aspectized_functions[id(obj)]
    del _applied_decorators[id(obj)]
