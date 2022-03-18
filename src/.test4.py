import sys

def f(x):
    return x**2





def test():
    def decorator(func):
        def wrapper(x):
            return func(x + 1)
        return wrapper
    setattr(sys.modules[__name__], "f", decorator(f))

test()
print(f(3))