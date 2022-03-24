__package__ = "tests"

def two():
    return 2


class Two:
    def __call__(self):
        return 2


def three():
    return 3


class Three:
    def three(self):
        return 3
