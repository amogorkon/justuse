from hashlib import sha256
from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest

s = '\U000aba08'

h = sha256(s.encode("utf8")).hexdigest()
