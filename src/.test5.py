import use

mod = use(use.Path(".test4.py"))


def decorator(func):
    def wrapper(x):
        return func(x + 1)

    return wrapper


use.apply_aspect(mod, decorator)
use.apply_aspect(mod, decorator)

print(mod.test(3))
