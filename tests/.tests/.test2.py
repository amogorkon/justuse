import inspect

def f(a,b, *args, c=2, **kws):
    print(inspect.getargvalues(inspect.currentframe()))