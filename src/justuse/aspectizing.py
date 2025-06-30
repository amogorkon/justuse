import ast
import contextlib
import inspect
import re
import sys
from collections import defaultdict, deque, namedtuple
from collections.abc import Callable, Iterable, Sized
from functools import wraps
from logging import getLogger
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import Any, Deque
from weakref import WeakKeyDictionary

from . import config
from .messages import _web_aspectized_dry_run, _web_tinny_profiler
from .utils import assumption

log = getLogger(__name__)


# Use WeakKeyDictionary to avoid memory leaks: when the object is deleted, its entry is removed.
_applied_decorators: "WeakKeyDictionary[object, dict[str, Deque[Callable]]]" = (
    WeakKeyDictionary()
)
"to see which decorators are applied, in which order"
_aspectized_functions: "WeakKeyDictionary[object, dict[str, Deque[Callable]]]" = (
    WeakKeyDictionary()
)
"the actually decorated functions to undo aspectizing"


def show_aspects():
    """Open a browser to properly display all the things that have been aspectized thus far."""
    print("decorators:", {k: v for k, v in _applied_decorators.items()})
    print("functions:", {k: v for k, v in _aspectized_functions.items()})
    # _web_aspectized(_applied_decorators, _aspectized_functions)


def is_callable(thing):
    try:
        return callable(
            object.__getattribute__(type(thing), "__call__")
        )  # to even catch weird non-callable __call__ methods
    except AttributeError:
        return False


HIT = namedtuple("Hit", "qualname name type success exception dunder mod_name")


def really_callable(thing):
    try:
        thing.__get__
        thing.__call__
        return True
    except Exception:
        return False


def apply_aspect(
    thing: object | Iterable[object],
    decorator: Callable,
    /,
    *,
    check: Callable = is_callable,
    dry_run: bool = False,
    pattern: str = "",
    excluded_names: set[str] | None = None,
    excluded_types: set[type] | None = None,
    file=None,
) -> None:
    """Apply the aspect as a side-effect, no copy is created."""
    regex = re.compile(pattern, re.DOTALL)

    if excluded_names is None:
        excluded_names = set()
    if excluded_types is None:
        excluded_types = set()

    visited = {id(obj) for obj in vars(object).values()}
    visited.add(id(type))

    hits = []

    def aspectize(
        thing: Any,
        decorator: Callable,
        /,
        *,
        qualname_lst: list[str] | None = None,
        mod_name: str,
    ) -> Iterable[HIT]:
        name = getattr(thing, "__name__", str(thing))
        if not qualname_lst:
            qualname_lst = []
        if id(thing) in visited:
            return

        qualname_lst.append(name)

        # let's stick within the module boundary
        try:
            thing_mod_name = inspect.getmodule(thing).__name__
        except AttributeError:
            thing_mod_name = "<unknown>"
        if thing_mod_name != mod_name:
            return

        for name in dir(thing):
            qualname = (
                ""
                if len(qualname_lst) < 3
                else "..." + ".".join(qualname_lst[-3:]) + "." + name
            )
            obj = getattr(thing, name, None)
            qualname = getattr(obj, "__qualname__", qualname)

            if obj is None:
                continue
            # Time to get serious!

            if type(obj) == type:
                aspectize(obj, decorator, qualname_lst=qualname_lst, mod_name=mod_name)

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
                if file:
                    print(msg, file=file)
            assert assumption(name, str) and len(name) > 0
            assert assumption(mod_name, str) and mod_name != ""

            hits.append(
                HIT(
                    qualname,
                    name,
                    type(wrapped),
                    success,
                    msg,
                    name.startswith("__") and name.endswith("__"),
                    mod_name,
                )
            )
            visited.add(id(wrapped))

    def call(m):
        try:
            mod_name = inspect.getmodule(m).__name__
        except AttributeError:
            mod_name = "<unknown>"
        aspectize(m, decorator, mod_name=mod_name)

    if isinstance(thing, Iterable):
        for x in thing:
            call(x)
    else:
        call(thing)

    if dry_run:
        if config.no_browser:
            print(
                "Tried to do a dry run and display the results, but no_browser is set in config."
            )
        else:
            print(
                "Please check your browser to select options and filters for aspects."
            )
            log.warning(
                "Please check your browser to select options and filters for aspects."
            )
            _web_aspectized_dry_run(
                decorator=decorator,
                pattern=pattern,
                check=check,
                hits=hits,
                mod_name=str(thing),
            )


def _wrap(*, thing: Any, obj: Any, decorator: Callable, name: str) -> Any:
    wrapped = decorator(obj)
    # Use WeakKeyDictionary to store per-object dicts of name -> deque
    if thing not in _applied_decorators:
        _applied_decorators[thing] = {}
    if name not in _applied_decorators[thing]:
        _applied_decorators[thing][name] = deque()
    _applied_decorators[thing][name].append(decorator)

    if thing not in _aspectized_functions:
        _aspectized_functions[thing] = {}
    if name not in _aspectized_functions[thing]:
        _aspectized_functions[thing][name] = deque()
    _aspectized_functions[thing][name].append(obj)

    # This will fail with TypeError on built-in/extension types.
    # We handle exceptions outside, let's not betray ourselves.
    setattr(thing, name, wrapped)
    return wrapped


def apply(*, thing, decorator, name):
    """Public version of aspectizing a single thing."""
    obj = getattr(thing, name)
    wrapped = decorator(obj)
    setattr(thing, name, wrapped)
    return wrapped


def _unwrap(*, thing: Any, name: str):
    try:
        original = _aspectized_functions[thing][name].pop()
        if not _aspectized_functions[thing][name]:
            del _aspectized_functions[thing][name]
        if not _aspectized_functions[thing]:
            del _aspectized_functions[thing]
    except (KeyError, IndexError):
        original = getattr(thing, name)

    with contextlib.suppress(KeyError, IndexError):
        _applied_decorators[thing][name].pop()
        if not _applied_decorators[thing][name]:
            del _applied_decorators[thing][name]
        if not _applied_decorators[thing]:
            del _applied_decorators[thing]
    setattr(thing, name, original)
    return original


def _qualname(thing):
    if type(thing) is bool:
        return str(thing)
    module = getattr(thing, "__module__", None) or getattr(
        thing.__class__, "__module__", None
    )
    qualname = (
        getattr(thing, "__qualname__", None)
        or getattr(thing, "__name__", None)
        or f"{type(thing).__name__}()"
    )
    qualname = destringified(qualname)
    return qualname if module == "builtins" else f"{module}::{qualname}"


def destringified(thing):
    return str(thing).replace("'", "")


def describe(thing):
    if thing is None:
        return "None"
    if type(thing) is bool:
        return str(thing)
    if isinstance(thing, (Sized)):
        if len(thing) == 0:
            return f"{_qualname(type(thing))} (empty)"
        if len(thing) < 4:
            return f"{_qualname(type(thing))}{destringified([_qualname(x) for x in thing])}]"
        else:
            return f"{_qualname(type(thing))}[{_qualname(type(thing[0]))}] ({len(thing)} items)"
    if isinstance(thing, (Iterable)):
        return f"{_qualname(type(thing))} (content indeterminable)"
    return _qualname(thing)


def woody_logger(thing: Callable) -> Callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """
    # A class is an instance of type - its type is also type, but only checking for `type(thing) is type`
    # would exclude metaclasses other than type - not complicated at all.
    name = _qualname(thing)

    if isinstance(thing, type):
        # wrapping the class means wrapping `__new__` mainly, which must return the original class, not just a proxy -
        # because a proxy never could hold up under scrutiny of type(), which can't be faked (like, at all).
        class wrapper(thing.__class__):
            def __new__(cls, *args, **kwargs):
                # let's check who's calling us
                caller = inspect.currentframe().f_back.f_code.co_name
                print(f"{caller}({args} {kwargs}) -> {name}()")
                before = perf_counter_ns()
                res = thing(*args, **kwargs)
                after = perf_counter_ns()
                print(
                    f"-> {name}() (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {type(res)}",
                    sep="\n",
                )
                return res

    else:
        # Wrapping a function is way easier.. except..
        @wraps(thing)
        def wrapper(*args, **kwargs):
            # let's check who's calling us
            caller = inspect.currentframe().f_back.f_code
            if caller.co_name == "<module>":
                caller = Path(caller.co_filename).name
            else:
                caller = f"{Path(caller.co_filename).name}::{caller.co_name}"
            print(
                f"{caller}([{', '.join(describe(a) for a in args)}] {destringified({k: describe(v) for k, v in kwargs.items()}) if len(kwargs) < 4 else f'dict of len{len(kwargs)}'}) -> {_qualname(thing)}"
            )
            before = perf_counter_ns()
            res = thing(*args, **kwargs)
            after = perf_counter_ns()
            if isinstance(res, (Sized)):
                print(
                    f"-> {describe(thing)} (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {describe(res)}",
                    sep="\n",
                )
                return res
            if isinstance(res, (Iterable)):
                print(
                    f"-> {describe(thing)} (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {describe(res)}",
                    sep="\n",
                )
                return res

            print(
                f"-> {describe(thing)} (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {describe(res)}",
                sep="\n",
            )
            return res

    return wrapper


_timings: dict[int, Deque[int]] = defaultdict(lambda: deque(maxlen=10000))


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


def _is_builtin(name, mod):
    if name in sys.builtin_module_names:
        return True

    if hasattr(mod, "__file__"):
        relpath = Path(mod.__file__).parent.relative_to(
            (Path(sys.executable).parent / "lib")
        )
        if relpath == Path():
            return True
        if relpath.parts[0] == "site-packages":
            return False
    return True


def _get_imports_from_module(mod):
    if not hasattr(mod, "__file__"):
        return
    with open(mod.__file__, "rb") as file:
        with contextlib.suppress(ValueError):
            for x in ast.walk(ast.parse(file.read())):
                if isinstance(x, ast.Import):
                    name = x.names[0].name
                    if (mod := sys.modules.get(name)) and not _is_builtin(name, mod):
                        yield name, mod
                if isinstance(x, ast.ImportFrom):
                    name = x.module
                    if (mod := sys.modules.get(name)) and not _is_builtin(name, mod):
                        yield name, mod


def iter_submodules(mod: ModuleType, visited=None, results=None) -> set[ModuleType]:
    """Find all modules recursively that were imported as dependency from  the given module."""
    if results is None:
        results = set()
    if visited is None:
        visited = set()
    for name, x in _get_imports_from_module(mod):
        if name in visited:
            continue
        visited.add(name)
        results.add(name)
        for name, x in _get_imports_from_module(sys.modules[name]):
            results.add(x)
            iter_submodules(x, visited, results)
    return results
