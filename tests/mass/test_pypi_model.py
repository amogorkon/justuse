import sys
import json
import traceback

import requests

from pypi_model import Package

with open("pypi.json", "r") as f:
    packages = json.load(f)


for package in packages["data"]:
    r = requests.get(f"https://pypi.org/pypi/{package['name']}/json")
    try:
        Package(**r.json())
    except:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        fail = {
            "name": package["name"],
            "err": {"type": str(exc_type), "value": str(exc_value), "traceback": tb.split("\n"),},
        }

        print(json.dumps(fail, indent=2, sort_keys=True))
        break

