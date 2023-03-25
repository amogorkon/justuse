import sys

import numpy as np

#for name, mod in list(sys.modules.items()):
#    del sys.modules[name]
#    sys.modules[f"'numpy'.name.removeprefix('numpy.')"] = mod

for x in sys.modules:
    print(x)