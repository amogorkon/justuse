from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest
from hashlib import sha256
from string import ascii_lowercase, digits

from itertools import combinations


results = {}

combs = combinations(digits + ascii_lowercase, 2)
for x in (''.join(y) for y in combs):
    text = x
    sha = sha256(text.encode("utf-8")).hexdigest()
    results[x] = sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))

combs = combinations(digits + ascii_lowercase, 3)
for x in (''.join(y) for y in combs):
    text = x
    sha = sha256(text.encode("utf-8")).hexdigest()
    results[x] = sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))

combs = combinations(digits + ascii_lowercase, 4)
for x in (''.join(y) for y in combs):
    text = x
    sha = sha256(text.encode("utf-8")).hexdigest()
    results[x] = sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))


for k,v in results.items():
    if not v:
        print(k)
