from functools import singledispatch, update_wrapper

def methdispatch(func):
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


from pathlib import Path

class Use:
    @methdispatch
    def __call__(self, x, a):
        print("can't handle!")
        
    @__call__.register(Path)
    def call_with_path(self, x, a,b, c):
        print("path")
        
    @__call__.register(str)
    def call_with_str(self, x, a, b):
        print(x)
        
use = Use()
use("adsf")