import inspect, ast, dis


def decorator(func):
    def wrapper(x):
        return func(x + 1)
    return wrapper


def test(x):
    frame = inspect.currentframe()
    while True:
        if frame.f_code == a.__code__:
            print(frame.f_code == a.__code__)
            break
        else:
            frame = frame.f_back

    return x**2


def a():
    test(2)
    
a()


