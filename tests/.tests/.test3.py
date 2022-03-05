import requests
from pprint import pprint

d = requests.get("https://pypi.org/pypi/pygame/2.1.2/json")


for release in d.json()["2.1.2"]:
    pprint(release)