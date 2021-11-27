def foo(x):
    return x**2

def decorator(func):
    def wrapper(*args, **kwargs):
        print(func, args, kwargs)
        return func(*args, **kwargs)
    return wrapper

_decorated_funcs = defaultdict(lambda: [])
_applied_decorators = defaultdict(lambda: [])

def apply_decorator(func, decorator):
    _decorated_funcs[func.__qualname__].append(func) # to undo
    _applied_decorators[func.__qualname__].append(decorator)
    func = decorator(func)

apply_decorator(func, decorator)