import os
from importlib import Path

here = Path(".").resolve()
os.chdir(Path("../src"))

import use

os.chdir(here)
