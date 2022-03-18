import sys

import numpy

import use
from use.aspectizing import _unwrap, _wrap


def f(x):
    return x**2


def test_decorate(reuse):
    def decorator(func):
        def wrapper(x):
            return func(x + 1)

        return wrapper

    _wrap(thing=sys.modules[__name__], obj=f, decorator=decorator, name="f")
    assert f(3) == 16
    _unwrap(thing=sys.modules[__name__], name="f")
    assert f(3) == 9


test_decorate(use)
# numpy @ use
mod = use(use.Path(".test4.py"))


use.apply_aspect(mod, use.woody_logger)
use.show_aspects()
print()
