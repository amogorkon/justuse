from justuse import Path, use

print("FROM", __file__)

use(Path("modB.py"))


def foo(x):
    return x * 2


use(Path("modA_test.py"), initial_globals=globals())
