from functools import lru_cache

@lru_cache
def fib(n):
    if n in (0,1):
        return n
    return fib(n-1) + fib(n -2)

class Recurse(Exception):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

def recurse(*args, **kwargs):
    raise Recurse(*args, **kwargs)
        
def tail_recursive(f):
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except Recurse as r:
                args = r.args
                kwargs = r.kwargs
                continue
    return decorated

@tail_recursive
def factorial_tail(n, accumulator=1):
    if n == 0:
        return accumulator
    recurse(n-1, accumulator=accumulator*n)
    
@lru_cache
def factorial_lru(n):
    if n in (0, 1):
        return 1
    return factorial_lru(n-1) * n

#print(factorial_lru(100))
#print(factorial_tail(100))

from time import perf_counter_ns
from statistics import mean, stdev, median

results = []
for x in range(10000):
    before = perf_counter_ns()
    factorial_lru(200)
    foo = perf_counter_ns()-before
    results.append(foo)
print([f(results) for f in [min, max, median, mean, stdev]])

results = []
for x in range(10000):
    before = perf_counter_ns()
    factorial_tail(200)
    foo = perf_counter_ns()-before
    results.append(foo)
print([f(results) for f in [min, max, median, mean, stdev]])

