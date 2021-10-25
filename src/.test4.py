import sys

sys.path.insert(0, "")
import tests.unit_test

import use

use("example-pypi-package", version="0.1.0", modes=use.auto_install)
