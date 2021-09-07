from packaging.version import Version as pkv

class Version(pkv):
    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], Version):
            return args[0]
        else:
            instance = super(cls, Version).__new__(cls)
            return instance

    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if isinstance(versionstr, Version):
            return
        if not (versionstr or major or minor or patch):
            raise ValueError(
                "Version must be initialized with either a string or major, minor and patch"
            )
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        return super().__init__(versionstr)


v = Version(Version("1"))
print(v)