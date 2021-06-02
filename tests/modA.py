

import use
print(__file__)

use(use.Path("modB.py"))


def foo(x):
   return x*2

use(use.Path("test_modA.py"), initial_globals=globals())