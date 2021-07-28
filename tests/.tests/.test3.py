class Mode(bytes):
    def __or__(self, other):
        return bytes(map(lambda a, b: a | b, self, other))
