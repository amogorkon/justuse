from hashlib import sha256

from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest


def test_hash_alphabet():
    H = sha256("hello world".encode("utf-8")).hexdigest()
    assert H == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(H)))
