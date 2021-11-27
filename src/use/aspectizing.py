import re
from logging import getLogger
from typing import Any

log = getLogger(__name__)

from use import (AmbiguityWarning, _applied_decorators, _aspectized_functions,
                 modules_excluded_from_aspectizing,
                 packages_excluded_from_aspectizing)


def _apply_aspect(
    thing,
    check,
    pattern,
    decorator,
    aspectize_dunders=False,
    ignore_boundaries=False,
    recursive=True,
    force=False,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    for name, obj in thing.__dict__.items():
        # first let's exclude everything that doesn't belong to the module to avoid infinite recursion and other nasty stuff
        if getattr(obj, "__module__", None) != getattr(thing, "__module__", None) and not ignore_boundaries:
            log.debug(
                f"{obj} is skipped because {getattr(obj, '__module__', None)} and {thing} {getattr(thing, '__module__', None)} are not the same."
            )
            continue
        # we can't really tell where stuff is coming from if __package__ isn't set, so let's refuse the temptation to guess
        if (
            getattr(obj, "__package__", None) is None
            and getattr(thing, "__module__", None) is None
            and not ignore_boundaries
        ):
            raise AmbiguityWarning(
                f"""Package of {obj} and its parent {thing} are not defined, thus we can't reliably enforce module boundaries.
                If you want to fix this issue, make sure that obj.__package__ is set or if you want to ignore this issue,  
                use._apply_aspect with ignore_boundaries=True."""
            )
        # then there are things that we really shouldn't aspectize (up for the user to fill)
        if (
            getattr(obj, "__module__", None) in modules_excluded_from_aspectizing
            or getattr(obj, "__package__", None) in packages_excluded_from_aspectizing
        ):
            log.debug(f"{obj} is skipped because {obj.__module__} or {obj.__package__} are excluded. ")
            continue
        if not aspectize_dunders and name.startswith("__") and name.endswith("__"):
            log.debug(f"{obj} is skipped because it's a dunder")
            continue
        if check(obj) and re.match(pattern, name):
            if decorator in _applied_decorators[obj.__qualname__] and not force:
                raise RuntimeError(
                    f"Applied decorator {decorator} to {obj} multiple times! Ff you intend this behaviour, use _apply_aspect with force=True"
                )
            _applied_decorators[obj.__qualname__].append(decorator)
            _aspectized_functions[obj.__qualname__] = obj
            thing.__dict__[name] = decorator(obj)
            log.debug(
                f"Applied {decorator.__qualname__} to {obj.__module__}::{obj.__qualname__} [{obj.__class__.__qualname__}]"
            )
    return thing
