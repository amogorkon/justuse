import builtins
import inspect
import re
import sys
from collections import namedtuple
from collections.abc import Callable
from enum import Enum
from functools import wraps
from importlib.util import spec_from_loader
from logging import getLogger
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import Any, DefaultDict, Deque, Optional
from warnings import catch_warnings, filterwarnings

from beartype import beartype
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from icontract import require

log = getLogger(__name__)

from use import config
from use.messages import _web_aspectized, _web_aspectized_dry_run

_applied_decorators: dict[int, Deque[Callable]] = DefaultDict(Deque)
"{qualname: [callable]} - to see which decorators are applied, in which order"
_aspectized_functions: dict[int, Deque[Callable]] = DefaultDict(Deque)
"{qualname: [callable]} - the actually decorated functions to undo aspectizing"


def show_aspects():
    """Open a browser to properly display all the things that have been aspectized thus far."""
    _web_aspectized(_applied_decorators, _aspectized_functions)


def is_callable(thing):
    try:
        object.__getattribute__(thing, "__call__")
        return True
    except AttributeError:
        return False


HIT = namedtuple("Hit", "qualname name type success")


@beartype
def apply_aspect(
    thing: Any,
    decorator: Callable,
    /,
    *,
    check: Callable = is_callable,
    pattern: str = "",
    module_name: Optional[str] = None,
    regex: Optional[re.Pattern] = None,
    aspectize_dunders: bool = False,
    recursive: bool = True,
    level: int = 0,
    dry_run: bool = False,
    visited: Optional[set[int]] = None,
    excluded_names: Optional[set[str]] = None,
    excluded_types: Optional[set[type]] = None,
    hits: Optional[list[HIT]] = None,
    last: bool = True,  # and first
) -> Any:
    name = str(thing)
    """Apply the aspect as a side-effect, no copy is created."""
    if visited is None:
        visited = set()
    if id(thing) in visited:
        return

    # initial call
    if module_name is None:
        try:
            module_name = inspect.getmodule(thing).__name__
        except AttributeError:
            module_name = "<unknown>"
    else:  # recursive call - let's stick within the module boundary
        try:
            thing_module_name = inspect.getmodule(thing).__name__
        except AttributeError:
            thing_module_name = "<unknown>"
        if thing_module_name != module_name:
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

        if id(obj) in visited:
            continue

        # Time to get serious!
        visited.add(id(obj))

        # check main::ProxyModule.__matmul__ for defaults
        # We can't really filter by name if it doesn't have a qualname
        if (
            getattr(obj, "__qualname__", getattr(obj, "__name__", None)) in excluded_names
            or type(obj) in excluded_types
            or not check(obj)
            or not regex.match(name)
        ):
            continue

        qualname = getattr(obj, "__qualname__", None)

        try:
            if dry_run:
                decorator(obj)
                wrapped = id(obj)
                hits.append(HIT(qualname, name, type(obj), True))
            else:
                wrapped = _apply_decorator(
                    thing=thing,
                    obj=obj,
                    decorator=decorator,
                    name=name,
                )
            # Mustn't forget to track!
            visited.add(id(wrapped))

        # AttributeError: readonly attribute
        except AttributeError:
            hits.append(HIT(qualname, name, False))
        visited.add(id(wrapped))

        if recursive:
            apply_aspect(
                obj,
                decorator,
                check=check,
                pattern=pattern,
                module_name=module_name,
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

    # this the last thing in the original call, after all the recursion
    if last and dry_run:
        if not config.no_browser:
            print("Please check your browser to select options and filters for aspects.")
            _web_aspectized_dry_run(
                decorator=decorator,
                pattern=pattern,
                check=check,
                hits=hits,
                module_name=module_name,
            )
        else:
            print("Tried to do a dry run and display the results, but no_browser is set in config.")
    return thing


@beartype
def _apply_decorator(*, thing, obj, decorator, name):
    previous_object_id = id(obj)
    try:
        wrapped = decorator(obj)
    except:
        log.info(f"Failed to decorate {thing.__name__} with {decorator.__name__}")
        return thing
    new_object_id = id(wrapped)

    _applied_decorators[new_object_id].extend(_applied_decorators[previous_object_id])
    _applied_decorators[new_object_id].append(decorator)

    _aspectized_functions[new_object_id].extend(_aspectized_functions[previous_object_id])
    _aspectized_functions[new_object_id].append(obj)

    setattr(thing, name, decorator(obj))

    # cleanup
    del _applied_decorators[previous_object_id]
    del _aspectized_functions[previous_object_id]
    return wrapped


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


def woody_logger(func: callable) -> callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{args} {kwargs} -> {func.__module__}::{func.__qualname__}")
        before = perf_counter_ns()
        res = func(*args, **kwargs)
        after = perf_counter_ns()
        log.debug(
            f"{func.__module__}::{getattr(func, '__qualname__', func.__name__)}({args}, {kwargs}) -> {res} {type(res)}"
        )
        print(
            f"{getattr(func, '__qualname__', func.__name__)} -- in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {res} {type(res)}"
        )
        return res

    return wrapper


def tinny_profiler(func: callable) -> callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{args} {kwargs} -> {func.__module__}::{func.__qualname__}")
        before = perf_counter_ns()
        res = func(*args, **kwargs)
        after = perf_counter_ns()
        log.debug(
            f"{func.__module__}::{getattr(func, '__qualname__', func.__name__)}({args}, {kwargs}) -> {res} {type(res)}"
        )
        print(
            f"{getattr(func, '__qualname__', func.__name__)} -- in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {res} {type(res)}"
        )
        return res

    return wrapper
