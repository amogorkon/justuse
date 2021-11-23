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

# Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves.
# This is necessary to compare sys.version_info with Version and make some tests more elegant, amongst other things.


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
    python_tag: Optional[str] = ""
    abi_tag: Optional[str]
    platform_tag: Optional[str] = "any"
    ext: Optional[str]


class PyPI_Release(BaseModel):
    comment_text: str = None
    digests: dict[str, str] = None
    url: str = None
    ext: str = None
    packagetype: str = None
    distribution: str = None
    requires_python: str = None
    python_version: str = None
    python_tag: str = None
    platform_tag: str = None
    filename: str = None
    abi_tag: str = None
    yanked: bool = False
    version: Version

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
    author: str = None
    author_email: str = None
    bugtrack_url: str = None
    classifiers: list[str] = None
    description: str = None
    description_content_type: str = None
    docs_url: str = None
    download_url: str = None
    downloads: PyPI_Downloads = None
    home_page: str = None
    keywords: str = None
    license: str = None
    maintainer: str = None
    maintainer_email: str = None
    name: str = None
    package_name: str = None
    package_url: str = None
    platform: str = None
    project_url: str
    project_urls: dict[str, str] = None
    release_url: str = None
    requires_dist: list[str] = None
    requires_python: str = None
    summary: str = None
    version: str = None
    yanked: bool = False
    yanked_reason: str = None


class PyPI_URL(BaseModel):
    comment_text: str = None
    digests: dict[str, str] = None
    downloads: int = -1
    filename: str = None
    has_sig: bool = False
    md5_digest: str = None
    packagetype: str = None
    python_version: str = None
    requires_python: str = None
    size: int = -1
    upload_time: str = None
    upload_time_iso_8601: str = None
    url: str = None
    yanked: bool = False
    yanked_reason: str = None


class PyPI_Project(BaseModel):
    releases: dict[Version, list[PyPI_Release]] = None
    urls: list[PyPI_URL] = None
    last_serial: int = None
    info: PyPI_Info = None

    def __init__(self, *, releases, urls, info, **kwargs):

        for version in list(releases.keys()):
            if not isinstance(version, str):
                continue
            try:
                Version(version)
            except packaging.version.InvalidVersion:
                del releases[version]

        releases2 = {
            str(ver_str): [
                {
                    **(rel_info if isinstance(rel_info, dict) else rel_info.dict()),
                    "version": Version(str(ver_str)),
                }
                for rel_info in release_infos
            ]
            for ver_str, release_infos in releases.items()
        }
        super(PyPI_Project, self).__init__(releases=releases2, urls=urls, info=info, **kwargs)


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
    if ".tar" in filename:
        ext = filename[filename.index(".tar") :]
    else:
        ext = pp.name[len(pp.stem) + 1 :]
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

    python_version = None
    if python_tag:
        python_version = python_tag.replace("cp", "")[0] + "." + python_tag.replace("cp", "")[1:]
    return _delete_none(
        {
            "distribution": distribution,
            "version": version,
            "build_tag": build_tag,
            "python_tag": python_tag,
            "abi_tag": abi_tag,
            "platform_tag": platform_tag,
            "python_version": python_version,
            "ext": ext,
        }
    )
