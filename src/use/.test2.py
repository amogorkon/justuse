import imp, importlib

name = "foo"
spec = importlib.machinery.PathFinder.find_spec(name)
mod = imp.new_module(name)
mod.__dict__["a"] = 34
exec(compile(spec.loader.get_source(name), name, "exec"), mod.__dict__)
