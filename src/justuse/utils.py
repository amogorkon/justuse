"""
Configuration, constants, and utility functions for justuse.
"""

import os
import re
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .constants import ALL, Modes

VERBOSE = Modes.verbose
INC_DUNDER = Modes.include_dunder


# Home directory for justuse
sessionID = uuid4()
del uuid4

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))
try:
    home.mkdir(mode=0o755, parents=True, exist_ok=True)
except PermissionError:
    home = tempfile.mkdtemp(prefix="justuse_")


def fraction_of_day(now: datetime = None) -> float:
    if now is None:
        now = datetime.now(timezone.utc)
    return round(
        (
            now.hour / 24
            + now.minute / (24 * 60)
            + now.second / (24 * 60 * 60)
            + now.microsecond / (24 * 60 * 60 * 1000 * 1000)
        )
        * 1000,
        6,
    )


# Enums and flags


def excel_style_datetime(now: datetime) -> float:
    """
    Build a float representing the current time in the excel format.
    First 4 digits are the year, the next two are the month, the next two are the day followed
    by a decimal point, then time in fraction of the day.
    Args:
        now (datetime): datetime instance to be converted
    Returns:
        float: Excel style datetime
    """
    return int(f"{now.year:04d}{now.month:02d}{now.day:02d}") + round(
        (now.hour * 3600 + now.minute * 60 + now.second) / 86400, 6
    )


def _is_callable(thing):
    try:
        return callable(
            object.__getattribute__(type(thing), "__call__")
        )  # to even catch weird non-callable __call__ methods
    except AttributeError:
        return False


def apply(
    decorator: Callable,
    kind: ALL,
    /,
    check=_is_callable,
    pattern="",
    mode: Modes = None,
):
    if mode is None:
        mode = Modes.DEFAULT

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


def assumption(obj: Any, expected: type) -> bool:
    """
    Check if obj is an instance of expected type or any type in a union.
    Usage:
        assert assumption(a, int)
        assert assumption(b, str | float)
    """
    types = (
        expected.__args__
        if hasattr(expected, "__origin__")
        and expected.__origin__ is type(None).__class__
        else None
    )
    if types is None and hasattr(expected, "__args__"):
        types = expected.__args__
    if types is None:
        types = (expected,)
    for exp in types:
        if isinstance(obj, exp):
            return True
    msg = (
        f"Expected {expected}, instead got {type(obj).__name__} (value: {obj})"
        if len(types) == 1
        else f"Expected one of {types}, instead got {type(obj).__name__} (value: {obj})"
    )
    raise AssertionError(msg)
