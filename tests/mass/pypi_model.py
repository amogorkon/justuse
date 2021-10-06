from __future__ import annotations

from typing import List, Optional, Dict

from pydantic import BaseModel


class Downloads(BaseModel):
    last_day: int
    last_month: int
    last_week: int


class ProjectUrls(BaseModel):
    __root__: Dict[str, str]


class Info(BaseModel):
    author: Optional[str]
    author_email: Optional[str]
    bugtrack_url: Optional[str]
    classifiers: List[str]
    description: Optional[str]
    description_content_type: Optional[str]
    docs_url: Optional[str]
    download_url: Optional[str]
    downloads: Downloads
    home_page: Optional[str]
    keywords: Optional[str]
    license: Optional[str]
    maintainer: Optional[str]
    maintainer_email: Optional[str]
    name: str
    package_url: str
    platform: Optional[str]
    project_url: str
    project_urls: Optional[ProjectUrls]
    release_url: str
    requires_dist: Optional[List[str]]
    requires_python: Optional[str]
    summary: Optional[str]
    version: Optional[str]
    yanked: bool
    yanked_reason: Optional[str]


class Digests(BaseModel):
    md5: str
    sha256: str


class Release(BaseModel):
    comment_text: Optional[str]
    digests: Digests
    downloads: int
    filename: str
    has_sig: bool
    md5_digest: str
    packagetype: Optional[str]
    python_version: str
    requires_python: Optional[str]
    size: int
    upload_time: str
    upload_time_iso_8601: str
    url: str
    yanked: bool
    yanked_reason: Optional[str]


class Releases(BaseModel):
    __root__: Dict[str, List[Release]]


class Package(BaseModel):
    info: Info
    last_serial: int
    releases: Releases
    urls: List[Release]
