def test1(a: int) -> int:
    return a / 2


def _test2(a: int) -> int:
    return a / 2


class Test:
    def __init__(self, a: int) -> None:
        self.a = a

    def test3(self) -> int:
        return self.a / 2
