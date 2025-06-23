from hashlib import sha256

from hypothesis import assume, example, given
from hypothesis import strategies as st
from use.hash_alphabet import JACK_as_num, hexdigest_as_JACK, num_as_hexdigest


@given(st.text())
@example("1t")
def test_jack(text):
    assume(text.isprintable())
    sha = sha256(text.encode("utf-8")).hexdigest()
    assert sha == num_as_hexdigest(JACK_as_num(hexdigest_as_JACK(sha)))
