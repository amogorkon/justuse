import importlib

name = "sys"
spec = importlib.util.find_spec(name)
mod = spec.loader.create_module(spec)
spec.loader.exec_module(mod)

print(mod.path)
del (mod)
mod = spec.loader.create_module(spec)
spec.loader.exec_module(mod)