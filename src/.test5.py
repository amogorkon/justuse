print(0)
import use

print(1)
np = use("numpy")
print(2)
use.apply_aspect(np, use.is_callable, "", use.log_call, dry_run=True)
print(3)
np.array([1,2,3])