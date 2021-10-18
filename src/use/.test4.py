from hash_alphabet import CJK_as_num, hexdigest_as_CJK, reverse_alphabet
import hash_alphabet as alph
from hashlib import sha256, sha3_512

H = sha256("hello world".encode("utf8")).hexdigest()
K = hexdigest_as_CJK(H)
N = CJK_as_num(K)

print(H)
print(int(H, 16))
print(K)
print(N)
