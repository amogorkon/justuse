from hashlib import sha1
from time import time
from base64 import b64encode



m = sha1()
m.update(f"Pharmatech 1663345906.1270907 49166977".encode())
print(b64encode(m.hexdigest().encode()))
print(int(m.hexdigest(), 16))