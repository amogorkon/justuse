import numpy
from beartype import beartype

import use

use.apply_aspect(use.iter_submodules(numpy), use.woody_logger)
# use.apply_aspect(numpy, use.woody_logger)
numpy.zeros((3, 3))
