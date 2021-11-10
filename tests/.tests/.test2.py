from hashlib import sha256

print(sha256('\xc2\x80\xc3\x92'.encode("utf8")).hexdigest())