from collections.abc import Callable
from functools import wraps
from time import perf_counter_ns

def _qualname(thing):
    module = getattr(thing, "__module__", None) or getattr(thing.__class__, "__module__", None)
    qualname = getattr(thing, "__qualname__", None) or getattr(thing, "__name__", None) or str(thing)
    return f"{module}::{qualname}"

def woody_logger(thing: Callable) -> Callable:
    """
    Decorator to log/track/debug calls and results.

    Args:
        func (function): The function to decorate.
    Returns:
        function: The decorated callable.
    """
    if type(thing) is type:
        class wrapper(thing.__class__):
            def __new__(cls, *args, **kwargs):
                print(f"{args} {kwargs} -> {thing.__name__}()")
                before = perf_counter_ns()
                res = thing(*args, **kwargs)
                after = perf_counter_ns()
                print(
                    f"-> {thing.__name__}() (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {type(res)}"
                )
                return res
    else:
        @wraps(thing)
        def wrapper(*args, **kwargs):
            print(f"{args} {kwargs} -> {_qualname(func)}")
            before = perf_counter_ns()
            res = func(*args, **kwargs)
            after = perf_counter_ns()
            print(
                f"-> {_qualname(func)} (in {after - before} ns ({round((after - before) / 10**9, 5)} sec) -> {res} {type(res)}"
            )
            return res
    return wrapper


@woody_logger
class Test:
    def foo(self):
        print(1)
    
    def bar(self):
        print(2)

t = Test()
