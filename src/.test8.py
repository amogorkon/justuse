class Foo:
    _x = 10
    x = property(lambda self: self._x, lambda self, x: self._x)
    