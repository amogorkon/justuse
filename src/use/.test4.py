from hash_alphabet import alphabet as E
import hash_alphabet as alph
from hashlib import sha256

def represent_num_as_base(num, base):
    if num == 0:
        return [0]
    digits = []
    while num:
        digits.append(num % base)
        num //= base
    return digits[::-1]

print(''.join(E[c] for c in represent_num_as_base(int(sha256("hello world".encode("utf8")).hexdigest(), 16), len(E))))

