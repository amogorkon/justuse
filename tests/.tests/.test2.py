from pydantic import BaseModel
from typing import List, Dict
import requests, json



class PyPI_Info(BaseModel):
    author: str = None
    author_email: str = None
    bugtrack_url: str = None
    classifiers: List[str] = None
    description: str = None
    description_content_type: str = None
    docs_url: str = None
    download_url: str = None
    downloads: Dict[str, int] = None
    home_page: str = None
    keywords: str = None
    license: str = None
    maintainer: str = None
    maintainer_email: str = None
    name: str
    package_url: str
    platform: str = None
    project_url: str
    project_urls: Dict[str, str] = None
    release_url: str = None
    requires_dist: str = None
    requires_python: str = None
    summary: str = None
    version: str = None
    yanked: bool = False
    yanked_reason: str = None

class PyPI_Release(BaseModel):
    comment_text: str = ""
    digests: Dict[str, str] = {}
    download_url: str = ""


class PyPI_URL(BaseModel):
    comment_text: str = ""
    digests: Dict[str, str] = {}
    downloads: int = -1
    filename: str
    has_sig: bool = False
    md5_digest: str = ""
    packagetype: str = ""
    python_version: str = ""
    requires_python: str = ""
    size: int = -1
    upload_time_iso_8601: str = ""
    url: str = ""
    yanked: bool = False
    yanked_reason: str = None


class PyPI_Project(BaseModel):
    info: PyPI_Info
    releases: Dict[str, List[PyPI_Release]]
    urls: List[PyPI_URL]
    last_serial: int

class PyPI_Project_Release(BaseModel):
    info: PyPI_Info
    last_serial: int
    releases: Dict[str, List[PyPI_Release]]
    urls: List[PyPI_URL] = None

data = requests.get("https://pypi.org/pypi/justuse/0.5.0/json").json()
O = PyPI_Project(**data)

