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
from use.messages import _web_aspectized, _web_aspectized_dry_run, _web_tinny_profiler

# TODO: use an extra WeakKeyDict as watchdog for object deletions and trigger cleanup in these here
_applied_decorators: DefaultDict[tuple[object, str], Deque[Callable]] = DefaultDict(Deque)
"{qualname: [callable]} - to see which decorators are applied, in which order"
_aspectized_functions: DefaultDict[tuple[object, str], Deque[Callable]] = DefaultDict(Deque)
"{qualname: [callable]} - the actually decorated functions to undo aspectizing"


def show_aspects():
    """Open a browser to properly display all the things that have been aspectized thus far."""
    print("decorators:", _applied_decorators)
    print("functions:", _aspectized_functions)
    # _web_aspectized(_applied_decorators, _aspectized_functions)


def is_callable(thing):
    try:
        object.__getattribute__(thing, "__call__")
        return True
    except AttributeError:
        return False


HIT = namedtuple("Hit", "qualname name type success exception dunder")


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
    dry_run: bool = False,
    visited: Optional[set[int]] = None,
    excluded_names: Optional[set[str]] = None,
    excluded_types: Optional[set[type]] = None,
    hits: Optional[list[HIT]] = None,
    last: bool = True,  # and first
    qualname_lst: Optional[list] = None,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    name = getattr(thing, "__name__", str(thing))
    if not qualname_lst:
        qualname_lst = []

    # to prevent recursion into the depths of hell and beyond
    if visited is None:
        visited = {id(obj) for obj in vars(object).values()}
        visited.add(id(type))

    if id(thing) in visited:
        return

    qualname_lst.append(name)

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

    for name in dir(thing):
        qualname = "" if len(qualname_lst) < 3 else "..." + ".".join(qualname_lst[-3:]) + "." + name
        obj = getattr(thing, name, None)
        qualname = getattr(obj, "__qualname__", qualname)

        if obj is None:
            continue
        # Time to get serious!

        if recursive and type(obj) == type:
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
                dry_run=dry_run,
                hits=hits,
                last=False,
                qualname_lst=qualname_lst,
            )

        if id(obj) in visited:
            continue

        if (
            qualname in excluded_names
            or type(obj) in excluded_types
            or not check(obj)
            or not regex.match(name)
        ):
            continue

        msg = ""
        success = True

        try:
            wrapped = _wrap(thing=thing, obj=obj, decorator=decorator, name=name)
            if dry_run:
                _unwrap(thing=thing, name=name)
        except BaseException as exc:
            wrapped = obj
            success = False
            msg = str(exc)
        hits.append(
            HIT(qualname, name, type(wrapped), success, msg, name.startswith("__") and name.endswith("__"))
        )
        visited.add(id(wrapped))

    if last and dry_run:
        if config.no_browser:
            print("Tried to do a dry run and display the results, but no_browser is set in config.")
        else:
            print("Please check your browser to select options and filters for aspects.")
            _web_aspectized_dry_run(
                decorator=decorator,
                pattern=pattern,
                check=check,
                hits=hits,
                module_name=module_name,
            )
    return thing


@beartype
def _wrap(*, thing: Any, obj: Any, decorator: Callable, name: str) -> Any:
    wrapped = decorator(obj)
    _applied_decorators[(thing, name)].append(decorator)
    _aspectized_functions[(thing, name)].append(obj)

    # This will fail with TypeError on built-in/extension types.
    # We handle exceptions outside, let's not betray ourselves.
    setattr(thing, name, wrapped)
    return wrapped


@beartype
def _unwrap(*, thing: Any, name: str):
    try:
        original = _aspectized_functions[(thing, name)].pop()
    except IndexError:
        del _aspectized_functions[(thing, name)]
        original = getattr(thing, name)

    try:
        _applied_decorators[(thing, name)].pop()
    except IndexError:
        del _applied_decorators[(thing, name)]
    setattr(thing, name, original)

    return original


def woody_logger(func: Callable) -> Callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """
    qualname = getattr(func, "__qualname__", None) or getattr(func, "__name__", None) or str(func)
    module = getattr(func, "__module__", None) or getattr(func.__class__, "__module__", None)
    if module:
        module += "."

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"{args} {kwargs} -> {module}{qualname}")
        before = perf_counter_ns()
        res = func(*args, **kwargs)
        after = perf_counter_ns()
        # log.debug(
        #     f"{func.__module__}::{getattr(func, '__qualname__', func.__name__)}({args}, {kwargs}) -> {res} {type(res)}"
        # )
        print(
            f"-> {module}{qualname} (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {res} {type(res)}"
        )
        return res

    return wrapper


_timings: dict[int, Deque[int]] = DefaultDict(lambda: Deque(maxlen=10000))


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
        before = perf_counter_ns()
        res = func(*args, **kwargs)
        after = perf_counter_ns()
        _timings[func].append(after - before)
        return res

    return wrapper


def show_profiling():
    _web_tinny_profiler(_timings)
