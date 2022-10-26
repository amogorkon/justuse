from base64 import b64encode
from hashlib import blake2b
from json import dumps

print(b64encode(dumps({"type":"hello"}).encode()),
      x := b64encode(dumps({"payload": "world"}).encode()),
      b64encode(blake2b(x).hexdigest().encode()),
sep=".")

print(b64encode("print('hello to you, too')".encode()))