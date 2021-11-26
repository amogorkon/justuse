import use

print("Hello from A!")

use(use.Path("module_b.py"), initial_globals={"foo": 23})