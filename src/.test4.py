from hashlib import sha256

import use
from use.hash_alphabet import hexdigest_as_JACK, num_as_hexdigest, reverse_alphabet


def JACK_as_num(string: str):
    if isinstance(string, bytes):
        string = string.decode()
    s = "".join(string.split())
    return sum(len(reverse_alphabet) ** i * reverse_alphabet[x] for i, x in enumerate(reversed(s)))

s = b"a"
h = sha256(s).hexdigest()

print(h == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(h))))