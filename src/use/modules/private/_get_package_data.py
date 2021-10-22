from functools import cache


from ...pypi_model import PyPI_Project


@cache(maxsize=512, typed=True)
def _get_package_data(package_name) -> PyPI_Project:
    json_url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(json_url)
    if response.status_code == 404:
        raise ImportError(Message.pebkac_unsupported(package_name))
    elif response.status_code != 200:
        raise RuntimeWarning(Message.web_error(json_url, response))
    return PyPI_Project(**response.json())
