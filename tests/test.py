import os

from pathlib import Path

here = Path()
os.chdir(Path(__file__).resolve().parent / "../src/")

import use

mod = use("pip")

os.chdir(here)
