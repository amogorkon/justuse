# Pydantic model for the PyPi JSON API

# If this code is inside use.py, it causes all kinds of problems.

from __future__ import annotations
from typing import Dict
from typing import List

from pydantic import BaseModel
from version_stuff import Version


class PyPI_Release(BaseModel):
    comment_text: str = None
    digests: Dict[str, str] = None
    url: str = None


class PyPI_Downloads(BaseModel):
    last_day: int
    last_month: int
    last_week: int


class PyPI_Info(BaseModel):
    author: str = None
    author_email: str = None
    bugtrack_url: str = None
    classifiers: List[str] = None
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
    project_urls: Dict[str, str] = None
    release_url: str = None
    requires_dist: List[str] = None
    requires_python: str = None
    summary: str = None
    version: str = None
    yanked: bool = False
    yanked_reason: str = None


class PyPI_URL(BaseModel):
    comment_text: str = None
    digests: Dict[str, str] = None
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
    releases: Dict[Version, List[PyPI_Release]] = None
    urls: List[PyPI_URL] = None
    last_serial: int = None
    info: PyPI_Info = None
