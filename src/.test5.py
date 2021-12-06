print(0)
import use

print(1)
np = use("numpy")
print(2)
use.aspect(np, use.any_callable, "", use.woody_logger, dry_run=True)
print(3)
use.aspect(np, use.any_callable, "", use.woody_logger)
print(4)
print(np.array([1, 2, 3]))
