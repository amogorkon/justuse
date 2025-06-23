from justuse import Path, use

print("FROM", __file__)
use(Path("tests_subdir/modC.py"))
