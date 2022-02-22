import requests

package_name = "example-pypi-package"
version = "0.0.5"
json_url = f"https://pypi.org/pypi/{package_name}/{version}/json"
response = requests.get(json_url)