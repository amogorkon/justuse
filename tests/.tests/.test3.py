from packaging.version import Version

class Version(Version):
    def __iter__(self):
        yield from self.release

v = Version("1.2.3")
