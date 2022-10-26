from decimal import Decimal, getcontext

def compute(n):
    getcontext().prec = n
    res = Decimal(0)
    for i in range(n):
        a = Decimal(1)/(16**i)
        b = Decimal(4)/(8*i+1)
        c = Decimal(2)/(8*i+4)
        d = Decimal(1)/(8*i+5)
        e = Decimal(1)/(8*i+6)
        r = a*(b-c-d-e)
        res += r
    return res

from time import perf_counter

for x in range(1,10000, 100):
    before = perf_counter()
    compute(x)
    after = perf_counter()
    print(after-before)
