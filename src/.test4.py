import sys

sys.path.insert(0, "")
import tests.unit_test

import use

use("example-pypi-package", modes=use.auto_install)
