from time import perf_counter_ns
from statistics import geometric_mean, stdev, median
from enum import Enum, IntEnum

from time import sleep
import numpy as np
import sys

from numba import jit, njit

from random import randint, random
from fuzzylogic.functions import gauss

from random import randint
from math import exp, sqrt

# random_vars = [randint(0, 100) for _ in range(10000000)]

def linear(m:float=0, b:float=0) -> callable:
    """A textbook linear function with y-axis section and gradient.
    
    f(x) = m*x + b
    BUT CLIPPED.

    >>> f = linear(1, -1)
    >>> f(-2)   # should be -3 but clipped
    0
    >>> f(0)    # should be -1 but clipped
    0
    >>> f(1)
    0
    >>> f(1.5)
    0.5
    >>> f(2)
    1
    >>> f(3)    # should be 2 but clipped
    1
    """
    def f(x) -> float:
        y = m * x + b
        if y <= 0:
            return 0
        elif y >= 1:
            return 1
        else:
            return y
    return f

def test_linear():
    f = linear(0, 100)
    return [f(x) for x in random_vars]

def linear_numba(m:float=0, b:float=0) -> callable:
    """A textbook linear function with y-axis section and gradient.
    
    f(x) = m*x + b
    BUT CLIPPED.

    >>> f = linear(1, -1)
    >>> f(-2)   # should be -3 but clipped
    0
    >>> f(0)    # should be -1 but clipped
    0
    >>> f(1)
    0
    >>> f(1.5)
    0.5
    >>> f(2)
    1
    >>> f(3)    # should be 2 but clipped
    1
    """
    @njit
    def f(x:float) -> float:
        y = m * x + b
        if y <= 0:
            return 0
        elif y >= 1:
            return 1
        else:
            return y
    return f

def test_linear_numba():
    f = linear_numba(0, 100)
    return [f(x) for x in random_vars]

def gauss_numba(c:float, b:float, *, c_m=1):
    """Defined by ae^(-b(x-x0)^2), a gaussian distribution.
    
    Basically a triangular sigmoid function, it comes close to human perception.

    vars
    ----
    c_m (a)
        defines the maximum y-value of the graph
    b
        defines the steepness
    c (x0)
        defines the symmetry center/peak of the graph
    """
    assert 0 < c_m <= 1
    assert 0 < b, "b must be greater than 0"
    max_float = sqrt(sys.float_info.max)

    @njit
    def f(x):
        # instead of try-except OverflowError
        if (x-c) > max_float:
            return 0
        else:
            o = (x - c)**2
        return c_m * exp(-b * o)
    return f


def gauss(c:float, b:float, *, c_m=1):
    """Defined by ae^(-b(x-x0)^2), a gaussian distribution.
    
    Basically a triangular sigmoid function, it comes close to human perception.

    vars
    ----
    c_m (a)
        defines the maximum y-value of the graph
    b
        defines the steepness
    c (x0)
        defines the symmetry center/peak of the graph
    """
    assert 0 < c_m <= 1
    assert 0 < b, "b must be greater than 0"
    max_float = sqrt(sys.float_info.max)

    def f(x):
        # instead of try-except OverflowError
        if (x-c) > max_float:
            return 0
        else:
            o = (x - c)**2
        return c_m * exp(-b * o)
    return f

def gaussmf(x, mean, sigma):
    """
    Gaussian fuzzy membership function.
    Parameters
    ----------
    x : 1d array or iterable
        Independent variable.
    mean : float
        Gaussian parameter for center (mean) value.
    sigma : float
        Gaussian parameter for standard deviation.
    Returns
    -------
    y : 1d array
        Gaussian membership function for x.
    """
    return np.exp(-((x - mean)**2.) / (2 * sigma**2.))

def test_gauss_numba():
    f = gauss_numba(0, 10)
    return [f(x) for x in random_vars]

def test_gauss():
    f = gauss(0, 10)
    return [f(x) for x in random_vars]

def test_gaussmf():
    return [gaussmf(x, 0, 10) for x in random_vars]


test_funcs = [
    lambda: list({}),
    lambda: list({1: 2}),
    lambda: list({(1, 2, 3): 4}),
    lambda: [3, 3, 4],
    lambda: [],
    lambda: list({0, 1, 2, 3, ...}),
    lambda: list({3, 9}),
    lambda: list(set()),
    lambda: [],
    lambda: [1, 2, 1, 1],
    lambda: sleep(0.5),
]


from collections import defaultdict
import inspect


foo = defaultdict(int)
foo["hello"] = 1
foo["world"] = 2
def test_dict1():
    return dict(foo)

def test_dict2():
    return {**foo}

def test_bool1():
    return not not foo

def test_bool2():
    return bool(foo)


def test_tuple():
    return 3 in {1, 3}

def test_set():
    return 3 in {1,3}

def timeit(func):
    res = []
    for _ in range(10_000_000):
        before = perf_counter_ns()
        func()
        after = perf_counter_ns()
        res.append(after - before)
    print("####################")
    print(inspect.getsource(func))
    for f in (min, geometric_mean, median, stdev):
        print(f.__name__, f"{f(res)} ns ({round(f(res)/10**9,5)} s)")
    return res


test_funcs = [test_tuple, test_set]

for f in test_funcs:
    timeit(f)