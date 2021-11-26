"""
Pydantic model for the PyPI JSON API.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import packaging
from packaging.version import Version as PkgVersion
from pydantic import BaseModel
from logging import getLogger

log = getLogger(__name__)

# Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves.
# This is necessary to compare sys.version_info with Version and make some tests more elegant, amongst other things.

class BaseModel(BaseModel):
    def __repr__(self):
        return self.__class__.__name__

class Version(PkgVersion):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Version):
            return args[0]
        else:
            return super(cls, Version).__new__(cls)

    def __init__(self, versionobj: Optional[Union[PkgVersion, str]] = None, *, major=0, minor=0, patch=0):
        if isinstance(versionobj, Version):
            return

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


def _delete_none(a_dict: dict[str, object]) -> dict[str, object]:
    for k, v in tuple(a_dict.items()):
        if v is None or v == "":
            del a_dict[k]
    return a_dict


class JustUse_Info(BaseModel):
    distribution: Optional[str]
    version: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str]
    abi_tag: Optional[str]
    platform_tag: Optional[str]
    ext: Optional[str]


class PyPI_Release(BaseModel):
    digests: dict[str, str]
    url: str
    packagetype: str
    distribution: str
    requires_python: Optional[str]
    python_version: Optional[str]
    python_tag: Optional[str]
    platform_tag: str
    filename: str
    abi_tag: str
    yanked: bool
    version: Version
    distribution: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str]
    abi_tag: Optional[str]
    platform_tag: Optional[str]
    ext: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    @property
    def is_sdist(self):
        return (
            self.packagetype == "sdist" or self.python_version == "source" or self.justuse.abi_tag == "none"
        )

    @property
    def justuse(self) -> JustUse_Info:
        pp = Path(self.filename)
        if ".tar" in self.filename:
            ext = self.filename[self.filename.index(".tar") + 1 :]
        else:
            ext = pp.name[len(pp.stem) + 1 :]
        rest = pp.name[0 : -len(ext) - 1]

        not_dash = lambda name: f"(?P<{name}>[^-]+)"
        not_dash_with_int = lambda name: f"(?P<{name}>[0-9][^-]*)"
        match = re.match(
            f"{not_dash('distribution')}-{not_dash('version')}-?{not_dash_with_int('build_tag')}?-?{not_dash('python_tag')}?-?{not_dash('abi_tag')}?-?{not_dash('platform_tag')}?",
            rest,
        )
        if match:
            return JustUse_Info(**_delete_none(match.groupdict()), ext=ext)
        return JustUse_Info()


class PyPI_Downloads(BaseModel):
    last_day: int
    last_month: int
    last_week: int


class PyPI_Info(BaseModel):
    author: Optional[str]
    author_email: Optional[str]
    bugtrack_url: Optional[str]
    classifiers: Optional[list[str]]
    description: Optional[str]
    description_content_type: Optional[str]
    docs_url: Optional[str]
    download_url: Optional[str]
    downloads: Optional[PyPI_Downloads]
    home_page: Optional[str]
    keywords: Optional[str]
    license: Optional[str]
    maintainer: Optional[str]
    maintainer_email: Optional[str]
    name: str
    package_name: Optional[str]
    package_url: str
    platform: Optional[str]
    project_url: Optional[str]
    project_urls: Optional[dict[str, str]]
    release_url: Optional[str]
    requires_dist: Optional[list[str]]
    requires_python: Optional[str]
    summary: Optional[str]
    version: Optional[str]
    yanked: Optional[bool]
    yanked_reason: Optional[str]


class PyPI_URL(BaseModel):
    digests: dict[str, str]
    url: str
    packagetype: str
    distribution: str
    requires_python: Optional[str]
    python_version: Optional[str]
    python_tag: Optional[str]
    platform_tag: str
    filename: str
    abi_tag: str
    yanked: bool
    version: Version
    distribution: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str]
    abi_tag: Optional[str]
    platform_tag: Optional[str]
    ext: Optional[str]


class PyPI_Project(BaseModel):
    releases: dict[Version, list[PyPI_Release]] = None
    urls: list[PyPI_URL] = None
    last_serial: int = None
    info: PyPI_Info = None

    def __init__(self, *, releases, urls, info, **kwargs):
      try:
        for version in list(releases.keys()):
            if not isinstance(version, str):
                continue
            try:
                Version(version)
            except packaging.version.InvalidVersion:
                del releases[version]

        def get_info(rel_info, ver_str):
           data = {
                    **rel_info,
                    **_parse_filename(rel_info["filename"]),
                    "version": Version(str(ver_str)),
                }
           if info.get("requires_python"): data["requires_python"] = info.get("requites_python")
           if info.get("requires_dist"): data["requires_dist"] = info.get("requires_dist")
           return data
        
        super(PyPI_Project, self).__init__(
          releases={
            str(ver_str): [
                get_info(rel_info, ver_str)
                for rel_info in release_infos
            ]
            for ver_str, release_infos in releases.items()
          },
          urls=[
            get_info(rel_info, ver_str) 
            for ver_str, rel_infos in releases.items()
            for rel_info in rel_infos
          ],
          info=info,
          **kwargs
        )
      finally:
        releases = None
        info = None
        urls = None
        


def _parse_filename(filename) -> dict:
    """
    REFERENCE IMPLEMENTATION - DO NOT USE
    Match the filename and return a dict of parts.
    >>> parse_filename("numpy-1.19.5-cp36-cp36m-macosx_10_9_x86_64.whl")
    {'distribution': 'numpy', 'version': '1.19.5', 'build_tag', 'python_tag': 'cp36', 'abi_tag': 'cp36m', 'platform_tag': 'macosx_10_9_x86_64', 'ext': 'whl'}
    """
    # Filename as API, seriously WTF...
    assert isinstance(filename, str)
    distribution = version = build_tag = python_tag = abi_tag = platform_tag = None
    pp = Path(filename)
    packagetype = None
    if ".tar" in filename:
        ext = filename[filename.index(".tar") :]
        packagetype = "source"
    else:
        ext = pp.name[len(pp.stem) + 1 :]
        packagetype = "bdist"
    rest = pp.name[0 : -len(ext) - 1]

    p = rest.split("-")
    np = len(p)
    if np == 5:
        distribution, version, python_tag, abi_tag, platform_tag = p
    elif np == 6:
        distribution, version, build_tag, python_tag, abi_tag, platform_tag = p
    elif np == 3:  # ['SQLAlchemy', '0.1.1', 'py2.4']
        distribution, version, python_tag = p
    elif np == 2:
        distribution, version = p
    else:
        return {}
    
    return {
            "distribution": distribution,
            "version": version,
            "build_tag": build_tag,
            "python_tag": python_tag,
            "abi_tag": abi_tag,
            "platform_tag": platform_tag,
            "ext": ext,
            "filename": filename,
            "packagetype": packagetype,
            "yanked_reason": "",
            "bugtrack_url": "",
            
    }
