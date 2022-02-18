
from random import choice, randint
from string import ascii_lowercase
d = {choice(ascii_lowercase)+str(x):randint(0, 100) for x in range(100000)}


def test1():
    return {k:d[k] for k in sorted(d)}

def test2():
    return dict(sorted(d.items()))

from time import time
from statistics import mean, median, stdev

print(len(d))
print(d)

timings = []
for _ in range(1000):
    before = time()
    test1()
    timings.append(time() - before)

print(*(f(timings) for f in (mean, median, stdev, min)))

timings = []
for _ in range(1000):
    before = time()
    test2()
    timings.append(time() - before)

print(*(f(timings) for f in (mean, median, stdev, min)))