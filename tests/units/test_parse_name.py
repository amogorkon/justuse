import pytest
from use.pimp import _parse_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("", (None, None)),
        ("foo", ("foo", "foo")),
        ("foo/bar", ("foo", "bar")),
        ("foo.py", ("foo", "foo.py")),
    ],
)
def test_parse_name(name, expected):
    assert _parse_name(name) == expected
