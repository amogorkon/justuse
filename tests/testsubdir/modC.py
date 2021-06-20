import inspect
print("FROM", __file__)


import use

# issue #8
from dataclasses import dataclass
@dataclass(init=False, repr=False)
class Stats:
    data_length: int  # total length of data
    peak_count = int  # number of detected peaks,
    

use(use.Path("../modD.py"))
