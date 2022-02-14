from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest, represent_num_as_base, AlphabetAccess
from hashlib import sha256
from string import ascii_lowercase, digits

from itertools import combinations


results = {}
bad_nums =  []
good_nums = []

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
    check = sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))
    results[x] = check
    if check:
        sha = sha256(text.encode("utf-8")).hexdigest()
        good_nums.append(represent_num_as_base(int(sha, 16), 28216))
    else:
        sha = sha256(text.encode("utf-8")).hexdigest()
        bad_nums.append(represent_num_as_base(int(sha, 16), 28216))


def foo(text):
    sha = sha256(text.encode("utf-8")).hexdigest()
    print(text, sha, represent_num_as_base(int(sha, 16), 28215)), 


for k,v in results.items():
    if not v:
        foo(k)