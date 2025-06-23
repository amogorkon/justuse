"""
Configuration, constants, and utility functions for justuse.
"""

import hashlib
import inspect
import os
import re
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from enum import Enum, Flag, IntEnum, auto
from pathlib import Path
from typing import Any
from uuid import uuid4

# Home directory for justuse
sessionID = uuid4()
del uuid4

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))
try:
    home.mkdir(mode=0o755, parents=True, exist_ok=True)
except PermissionError:
    home = tempfile.mkdtemp(prefix="justuse_")

# Version string for packaging and runtime
__version__ = "0.9.1.1.0"

# Utility: fraction of day


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
class Hash(Enum):
    sha256 = hashlib.sha256
    blake = hashlib.blake2s


class Modes(IntEnum):
    auto_install = 2**0
    fatal_exceptions = 2**1
    reloading = 2**2
    no_public_installation = 2**3
    fastfail = 2**4
    recklessness = 2**5
    no_browser = 2**6
    no_cleanup = 2**7


class ModeFlags(Flag):
    AUTO_INSTALL = auto()
    FATAL_EXCEPTIONS = auto()
    RELOADING = auto()
    FASTFAIL = auto()
    RECKLESS = auto()
    DEFAULT = auto()
    NO_PUBLIC_INSTALLATION = auto()
    NO_CLEANUP = auto()
    NO_BROWSER = auto()


(
    AUTO_INSTALL,
    FATAL_EXCEPTIONS,
    RELOADING,
    FASTFAIL,
    RECKLESS,
    DEFAULT,
    NO_PUBLIC_INSTALLATION,
    NO_CLEANUP,
    NO_BROWSER,
) = ModeFlags


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


class ALL(Enum):
    methods = inspect.ismethod
    properties = inspect.isdatadescriptor
    functions = inspect.isfunction
    classes = inspect.isclass


def apply(
    decorator: Callable,
    kind: ALL,
    /,
    check=_is_callable,
    pattern="",
    mode: ModeFlags = None,
):
    if mode is None:
        mode = DEFAULT

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
