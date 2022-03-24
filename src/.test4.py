from time import perf_counter_ns
from collections import deque, defaultdict
from functools import wraps
from typing import DefaultDict, Deque

from time import perf_counter_ns as time
from collections import deque, defaultdict

_timings: dict[int, Deque[int]] = DefaultDict(lambda: deque(maxlen=10))


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

@tinny_profiler
def a():
    print("adsf")
    
a()