

def deferred(**defaults):
    def func_wrapper(func):
        def wrapper(*args, **kwargs):
            print(func, args, kwargs)
            print(defaults)
            default_calls = {k: v() for k, v in defaults.items()}
            updated_kwargs = {k: v if v is not None else default_calls[k] for k,v in kwargs.items()}
            print(func, args, updated_kwargs)
            return func(*args, **updated_kwargs)
        return wrapper
    return func_wrapper

@deferred(b=list)
def f(a, b=None):
    print(b)
    return b.append(a)


    