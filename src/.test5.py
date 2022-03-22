import use
from beartype import beartype

mod = use("requests")
use.apply_aspect(mod, beartype)
use.apply_aspect(mod, use.tinny_profiler)
for _ in range(1000):
    mod.get("https://pypi.org/project/beartype/")
use.show_profiling()