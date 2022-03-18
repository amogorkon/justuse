import sys

import numpy

import use
from use.aspectizing import _unwrap, _wrap

numpy @ use
mod = use(use.Path(".test4.py"))


use.apply_aspect(mod, use.woody_logger)
use.show_aspects()
print()
