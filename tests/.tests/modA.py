import use

print("FROM", __file__)

use(use.Path("modB.py"))


def foo(x):
   return x*2

use(use.Path("modA_test.py"), initial_globals=globals())
