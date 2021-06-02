import inspect

print(__file__)


import use

from dataclasses import dataclass
@dataclass(init=False, repr=False)
class Stats:
    data_length: int  # total length of data
    peak_count = int  # number of detected peaks,
    
use(use.Path("../modD.py"))
