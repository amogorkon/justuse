from time import perf_counter_ns
from statistics import geometric_mean, stdev, median, mean
import inspect
import re, string
from random import seed, randint
from collections import Counter


print("RUNNING...")
##########################################

class Test1:
    __slots__ = "foo",

t1 = Test1()

def test0():
    t1.foo = 'foo'
    t1.foo
    del t1.foo

class Test2:
    pass

t2 = Test2()

def test1():
    t2.foo = 'foo'
    t2.foo
    del t2.foo

##########################################

test_funcs = [test0, test1,]
#test_funcs = []

def rel_stdev(res):
    return stdev(res)/mean(res)

def timeit(func):
    res = []
    for _ in range(1000000):  # https://en.wikipedia.org/wiki/Sample_size_determination#Tables
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