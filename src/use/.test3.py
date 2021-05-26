class Use:
    def __call__(self):
        return 34
    
    def __matmul__(self, other):
        print(other)

use = Use()

