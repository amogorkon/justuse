from time import perf_counter_ns
from statistics import geometric_mean, stdev, median, mean
import inspect
import re, string
from random import seed, randint
from collections import Counter


print("RUNNING...")
##########################################

from collections import deque

def test0():
    L = []
    for x in range(1000):
        L.insert(0, x)

def test1():
    D = deque()
    for x in range(1000):
        D.appendleft(x)

##########################################

test_funcs = [test0, test1]

def rel_stdev(res):
    return stdev(res)/mean(res)

def timeit(func):
    res = []
    for _ in range(100000):  # https://en.wikipedia.org/wiki/Sample_size_determination#Tables
        before = perf_counter_ns()
        func()
        after = perf_counter_ns()
        res.append(after - before)
    print("####################")
    print(inspect.getsource(func))
    print(len(res), "runs")
    for f in (min, geometric_mean, median, stdev, sum):
        print(f.__name__, f"{f(res)} ns ({round(f(res)/10**9,5)} s)")
    print(f"relative stdev {rel_stdev(res):.2%}")
    return res

for f in test_funcs:
    timeit(f)