import typing
from collections.abc import Sequence

from use.pimp import _check, _is_compatible


def test_same_signature_different_kwarg_order():
    def f1(a, b, *, y: float, x: str):
        return x

    def f2(a, b, *, x: str, y: float):
        return x

    assert _is_compatible(f1, f2)
    assert _is_compatible(f2, f1)


def test_float_to_int():
    def f3(x: float):
        return x

    def f4(x: int):
        return x

    assert _is_compatible(f3, f4)
    assert not _is_compatible(f4, f3)


def test_sequence_to_list():
    def f5(x: Sequence[int]):
        return x

    def f6(x: list[int]):
        return x

    def f7(x: typing.Sequence):
        return x

    def f8(x: Sequence):
        return x

    assert _is_compatible(f5, f6)
    assert not _is_compatible(f6, f5)
    # assert _is_compatible(f7, f8)
    # assert not _is_compatible(f8, f7)


def test_signature_compatibility():
    assert _check(list, typing.List[int])
    assert _check(Sequence[int], list[int])
    assert not _check(list, typing.Sequence)
    assert not _check(list, typing.Sequence[int])
    assert _check(list[int], typing.List[int])
    assert _check(list, list[int])
    assert not _check(list[int], Sequence[int])
    assert not _check(list[int], typing.Sequence)
    assert not _check(list[int], typing.Sequence[int])
    assert not _check(typing.List[int], Sequence[int])
    assert not _check(typing.List[int], typing.Sequence)
    assert not _check(typing.List[int], typing.Sequence[int])
    assert not _check(Sequence[int], typing.Sequence)
    assert _check(Sequence[int], typing.Sequence[int])
