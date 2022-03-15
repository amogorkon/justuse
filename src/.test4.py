from packaging.version import Version as PkgVersion
from typing import Optional, Union

class Version(PkgVersion):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Version):
            return args[0]
        else:
            return super(cls, Version).__new__(cls)

    def __init__(self, versionobj: Optional[Union[PkgVersion, str]] = None, *, major=0, minor=0, patch=0):
        if isinstance(versionobj, Version):
            return

        versionobj = '.'.join(versionobj.split('.'[:1]))  
        
        if versionobj:
            super(Version, self).__init__(versionobj)
            return

        if major is None or minor is None or patch is None:
            raise ValueError(
                f"Either 'Version' must be initialized with either a string, packaging.version.Verson, {__class__.__qualname__}, or else keyword arguments for 'major', 'minor' and 'patch' must be provided. Actual invocation was: {__class__.__qualname__}({versionobj!r}, {major=!r}, {minor=!r})"
            )

        # string as only argument
        # no way to construct a Version otherwise - WTF
        versionobj = ".".join(map(str, (major, minor, patch)))
        super(Version, self).__init__(versionobj)

    def __iter__(self):
        yield from self.release

    def __repr__(self):
        return f"Version('{super().__str__()}')"

    def __hash__(self):
        return hash(self._version)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        return Version(value)
    
Version('1.2.3.4.5')