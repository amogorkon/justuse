import use
from use.pydantics import PyPI_Project
import requests

d = requests.get("https://pypi.org/pypi/pygame/2.1.2/json")

proj = PyPI_Project(**d.json())