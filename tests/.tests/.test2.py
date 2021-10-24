from hashlib import sha256

print(len(sha256("hello world".encode("utf8")).hexdigest()))