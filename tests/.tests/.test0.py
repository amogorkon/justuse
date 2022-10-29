import requests
from pprint import pprint
from use import pydantics
from pydantic import BaseModel
from typing import Optional


package_name = "h5py"
json_url = 'https://pypi.org/pypi/h5py/json'
json_url = 'https://pypi.org/pypi/h5py/2.10.0/json'
reply = requests.get(json_url).json()

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
    distribution: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str]
    abi_tag: Optional[str]
    platform_tag: Optional[str]
    ext: Optional[str]
    
urls = [PyPI_URL(**R) for R in reply["urls"]]