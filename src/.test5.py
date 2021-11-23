from hashlib import sha256
from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest

s = b'\x9a\xe6\xa8'
h = sha256(s).hexdigest()

h == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(h)))