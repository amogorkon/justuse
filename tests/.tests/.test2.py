from beartype import beartype
from pathlib import Path
import os, sys

sys.path.insert(0, "")

cwd = Path("")
src = (Path(__file__).parent / ".." / ".." / "src" / "use").resolve()
print(src)
os.chdir(src)
from pypi_model import PyPI_Project, PyPI_Release, Version
os.chdir(cwd)

@beartype
def test(PyPI_Project):
    print("adsf")
    
test(PyPI_Release(version=Version("1")))