import requests
from pprint import pprint

owner = "amogorkon"
repo = "justuse"
url = f"https://api.github.com/repos/{owner}/{repo}/commits"
pprint(requests.get(url).json())