from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest
from hashlib import sha256
from string import ascii_letters


results = {}

for x in ascii_letters:
    text = x
    sha = sha256(text.encode("utf-8")).hexdigest()
    results[x] = sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))
