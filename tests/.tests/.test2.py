from packaging.version import Version as PkgVersion
class Version(PkgVersion):    
    def __init__(self, versionstr, *, major=0, minor=0, patch=0):
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        if isinstance(versionstr, str):
            return super().__init__(versionstr)
        else:
            return super().__init__(str(versionstr))  # this is just wrong :|

    def __iter__(self):
        yield from self.release
