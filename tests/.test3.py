import numba
import numpy as np
from time import monotonic
from random import sample

@numba.njit
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx

ChannelIDs = np.random.randint(0,100000, 100000, int)


results = {}

idxs = list(sample(list(ChannelIDs), 5))
print(idxs)

before_all = monotonic()
for x in idxs:
    timings = []
    for _ in range(100000):
        before = monotonic()
        res = np.where(ChannelIDs == x)[0]
        now = monotonic()
        timings.append(now - before)
    results[x] = max(timings)

print(monotonic() - before_all)
print(results)

results = {}
before_all = monotonic()

for x in idxs:
    timings = []
    for _ in range(100000):
        before = monotonic()
        res = index(ChannelIDs, x)[0]
        now = monotonic()
        timings.append(now - before)
    results[x] = max(timings)

print(monotonic() - before_all)
print(results)