class Test:
    def __matmul__(self, other):
        print("matmul", other)
        
t = Test()