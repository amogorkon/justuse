import requests
from use.pypi_model import PyPI_Project, PyPI_Release, PyPI_Info

package_name = "example-pypi-package"
version = "0.0.5"
json_url = f"https://pypi.org/pypi/{package_name}/json"
response = requests.get(json_url)
ppp = PyPI_Project(**response.json())