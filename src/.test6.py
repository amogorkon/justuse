from hashlib import sha256
from use.hash_alphabet import hexdigest_as_JACK, JACK_as_num, num_as_hexdigest, represent_num_as_base, AlphabetAccess


def represent_num_as_base(num, base):
    if num == 0:
        return [0]
    digits = []
    while num:
        digits.append(num % base)
        num //= base
    return digits[::-1]


X = represent_num_as_base(28215**2+28214, 28215)
print(X)
jack = ''.join(AlphabetAccess.alphabet[c] for c in X)
print(jack)