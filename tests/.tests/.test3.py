from hashlib import sha256

print(sha256("foo".encode("utf8")).hexdigest())
