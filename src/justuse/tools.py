"""
Module to hold the decorators and other utility functions used in justuse.
"""

import inspect
import re
from collections.abc import Callable
from enum import Enum, Flag, auto
from typing import Any


def _is_callable(thing):
    try:
        return callable(
            object.__getattribute__(type(thing), "__call__")
        )  # to even catch weird non-callable __call__ methods
    except AttributeError:
        return False


class ALL(Enum):
    methods = inspect.ismethod
    properties = inspect.isdatadescriptor
    functions = inspect.isfunction
    classes = inspect.isclass


class ModeFlags(Flag):
    RECURSIVE = auto()
    VERBOSE = auto()
    TRIAL = auto()
    INC_DUNDER = auto()
    DEFAULT = auto()


RECURSIVE, VERBOSE, TRIAL, INC_DUNDER, DEFAULT = ModeFlags


def apply(
    decorator: Callable,
    kind: ALL,
    /,
    check=_is_callable,
    pattern="",
    mode: ModeFlags = DEFAULT,
):
    def sugar(*things: list[Any]):
        visited = {id(obj) for obj in vars(object).values()}
        visited.add(id(type))
        for thing in things:
            if id(thing) in visited:
                continue
            for name in dir(thing):
                obj = getattr(thing, name, None)
                qualname = getattr(obj, "__qualname__", name)

                if (
                    obj is None
                    or id(obj) in visited
                    or not check(obj)
                    or not kind(obj)
                    or not re.match(pattern, name)
                    or (
                        INC_DUNDER in mode
                        and name.startswith("__")
                        and name.endswith("__")
                    )
                ):
                    continue

                try:
                    wrapped = decorator(obj)
                    setattr(thing, name, wrapped)
                except BaseException as exc:
                    if VERBOSE in mode:
                        print(
                            f"Failed to apply decorator {decorator} to {qualname}: {exc}"
                        )
                    wrapped = obj
                visited.add(id(wrapped))
                if VERBOSE in mode:
                    print(f"Applied decorator {decorator} to {qualname}")
        return things[0] if len(things) == 1 else things

    return sugar
