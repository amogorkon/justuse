# Pydantic model for the PyPi JSON API

# If this code is inside use.py, it causes all kinds of problems.

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path
import re

from pydantic import BaseModel

from packaging.version import Version as PkgVersion

#%% Version and Packaging

# Well, apparently they refuse to make Version iterable, so we'll have to do it ourselves.
# This is necessary to compare sys.version_info with Version and make some tests more elegant, amongst other things.


class Version(PkgVersion):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Version):
            return args[0]
        else:
            return super(cls, Version).__new__(cls)

    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if isinstance(versionstr, Version):
            return
        if not (versionstr or major or minor or patch):
            raise ValueError("Version must be initialized with either a string or major, minor and patch")
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        return super().__init__(versionstr)

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


class QuietModel(BaseModel):
    def __repr__(self):
        return "%s()" % type(self).__qualname__


class JustUse_Info(BaseModel):
    distribution: Optional[str]
    version: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str] = ""
    abi_tag: Optional[str]
    platform_tag: Optional[str] = "any"
    ext: Optional[str]


class PyPI_Release(QuietModel):
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
        return self.packagetype == "sdist" or self.python_version == "source" or self.justuse.abi_tag == "none"

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


class PyPI_Downloads(QuietModel):
    last_day: int
    last_month: int
    last_week: int


class PyPI_Info(QuietModel):
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


class PyPI_URL(QuietModel):
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


class PyPI_Project(QuietModel):
    releases: dict[Version, list[PyPI_Release]] = None
    urls: list[PyPI_URL] = None
    last_serial: int = None
    info: PyPI_Info = None

    def sort_releases_by_install_method(self):
        return PyPI_Project(
            **{
                **self.dict(),
                **{
                    "releases": {
                        k: [x.dict() for x in sorted(v, key=lambda r: r.is_sdist)] for k, v in self.releases.items()
                    }
                },
            }
        )

    def recommend_best_version(self):
        sorted(self.releases.keys(), reverse=True)

    def __init__(self, **kwargs):

        for version in list(kwargs["releases"].keys()):
            try:
                Version(version)
            except:
                del kwargs["releases"][version]

        kwargs["releases"] = {k: [{**_v, "version": Version(k)} for _v in v] for k, v in kwargs["releases"].items()}

        super().__init__(**kwargs)
