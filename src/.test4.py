hashes = {("a", "aa", "aaa"), ("b", "bb", "bbb")}
name = "pygame"
version = "1.2.3"

hashes = ["adsf adf"]

def is_JACK(H):
    return True

def JACK_as_num(H):
    return H

hashes: set[int] = {
    JACK_as_num(H) if is_JACK(H) else int(H, 16) for H in ("".join(H.split()) for H in hashes)
}
