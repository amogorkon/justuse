import builtins
import re
import sys
from collections import defaultdict, deque
from importlib.util import spec_from_loader
from inspect import getmembers, isclass
from logging import getLogger
from types import ModuleType
from typing import Any, Callable
from warnings import warn, catch_warnings, filterwarnings

from beartype.roar import BeartypeDecorHintPep585DeprecationWarning, BeartypeCallHintPepParamException
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
    visited: set = None,
) -> Any:
    """Apply the aspect as a side-effect, no copy is created."""
    if visited is None:
        visited = set()
    regex = re.compile(pattern, re.DOTALL)
    # object.__dir__ is the *only* reliable way to get all attributes 
    # of an object that nobody can override,
    # dir() and inspect.getmembers both build on __dir__, but are not
    # reliable.
    for name in object.__dir__(thing):
        if name in (
            "__init_subclass__",
            "__new__",
            "__prepare__",
            "__module__",
            "__matmul__",
        ):
            continue
        from use.main import ProxyModule
        try:
            obj = object.__getattribute__(thing, name)
        except AttributeError:
            try:
                obj = getattr(
                    object.__getattribute__(
                       thing, "_ProxyModule__implementation"
                    ),
                    name
                )
            except AttributeError:
                continue
        
        if id(obj) in visited:
            continue
        if type(obj) is ProxyModule:
            continue
        visited.add(id(obj))
        
        if isinstance(obj, ModuleType):
            if recursive and obj.__name__.startswith(thing.__name__):
                log.debug(
                    f"{str(obj)[:20]} is being aspectized recursively"
                )
                # Aspectize the new module
                mod = obj
            else:
                continue
        else:
            if not check(obj) or not regex.match(name):
                continue
            try:
                mod = lookup_module(obj)
            except:
                pass
        if mod is None: mod = obj
        orig_obj = None
        try:
            # then there are things that we really shouldn't aspectize 
            # (up for the user to fill)
            module_name, loader, spec = get_module_info(mod)
            ispackage, package_name = get_package(mod)
    
            if module_name in modules_excluded_from_aspectizing:
                log.debug(
                    f"{str(obj)[:20]} [{type(obj)}] is skipped "
                    f"because {module_name} is excluded."
                )
                continue
            if package_name in packages_excluded_from_aspectizing:
                log.debug(
                    f"{str(obj)[:20]} [{type(obj)}] is skipped "
                    f"because {package_name} is excluded."
                )
                continue
            if (not aspectize_dunders 
                and name.startswith("__")
                and name.endswith("__")
            ):
                log.debug(
                    f"{str(obj)[:20]} [{type(obj)}] is skipped "
                    f"because it's a dunder"
                )
                continue
            
            if not isinstance(obj, ModuleType):
                if isclass(obj):
                    orig_obj = obj
                    obj = object.__getattribute__(obj, "__call__")
                previous_object_id = id(obj)
                wrapped = decorator(obj)
                new_object_id = id(wrapped)
    
                _applied_decorators[new_object_id].extend(
                    _applied_decorators[previous_object_id]
                )
                _applied_decorators[new_object_id].append(decorator)
    
                _aspectized_functions[new_object_id].extend(
                    _aspectized_functions[previous_object_id]
                )
                _aspectized_functions[new_object_id].append(obj)
    
                setattr(thing, name, decorator(obj))
    
                # cleanup
                del _applied_decorators[previous_object_id]
                del _aspectized_functions[previous_object_id]
        except (AttributeError, TypeError, BeartypeCallHintPepParamException):
            continue
        log.debug(
            f"{decorator.__qualname__} @ "
            f"{module_name}::"
            f"{obj.__dict__.get('__qualname__',obj.__dict__.get('__name__',''))} "
            f"[{obj.__class__.__name__}]"
        )
        if recursive:
            log.debug(
                f"recursing on {str(obj)[:20]} [{type(obj)}] "
                f"with ({check}, {pattern} {decorator.__qualname__}"
            )
            _apply_aspect(
                orig_obj or obj,
                check=check,
                pattern=pattern,
                decorator=decorator,
                aspectize_dunders=aspectize_dunders,
                ignore_boundaries=ignore_boundaries,
                recursive=True,
                force=force,
                visited=visited,
            )
            log.debug("finished recursion on %s", obj)
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
    elif (
            hasattr(object, "__init__") 
        and hasattr(object.__init__, "__module__")
    ):
        object_mod = sys.modules[object.__init__.__module__]
    elif not isinstance(object, type):
        object_mod = builtins
    if not object_mod:
        raise NotImplementedError(
            "Don't know how to find module for '%s.%s' objects" % (
                object.__module__, object.__qualname__
            )
        )
    return object_mod


# @lru_cache(maxsize=0)
def get_module_info(mod: ModuleType):
    module_name = getattr(mod, "__name__", None) or mod.__spec__.name
    loader = getattr(mod, "__loader__", None) or mod.__spec__.loader
    spec = (
        getattr(mod, "__spec__", None) 
        or spec_from_loader(module_name, loader)
    )
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
        package_name = (
            getattr(mod, "__package__", None) 
            or ".".join(module_parts[:-1])
        )
    return is_package, package_name


def isbeartypeable(thing):
    with catch_warnings():
        filterwarnings(
            action="ignore",
            category=BeartypeDecorHintPep585DeprecationWarning
        )
        if type(thing).__name__.startswith("builtin_method"):
            return False
        if thing is type or thing is object or not hasattr(thing, "__dict__"):
            return False
    
        try:
            beartype(thing)
            return True
        except:
            return False