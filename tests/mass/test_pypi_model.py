import json
import sys
import traceback

import requests

from pypi_model import PyPI_Project

with open("pypi.json", "r") as f:
    packages = json.load(f)


for i, pkg in enumerate(packages["data"]):
    r = requests.get(f"https://pypi.org/pypi/{pkg['name']}/json")
    try:
        project = PyPI_Project(**r.json())
        filtered = project.filter_by_version_and_current_platform(project.info.version)
        for version in filtered.releases.values():
            for release in version:
                print(release.filename)
                print(release.justuse)
    except:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        fail = {
            "name": pkg["name"],
            "err": {
                "type": str(exc_type),
                "value": str(exc_value),
                "traceback": tb.split("\n"),
            },
        }

        print(json.dumps(fail, indent=2, sort_keys=True))
        break

    if i > 4:
        break
