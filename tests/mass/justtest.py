import sys
import json
from typing import List, Optional
import re
import logging
import traceback

import use
from pydantic import BaseModel

logging.root.setLevel(logging.ERROR)


class PackageToTest(BaseModel):
    name: str
    versions: List[str]
    repo: Optional[str] = None
    stars: Optional[int] = None


class Packages(BaseModel):
    data: List[PackageToTest] = []

    def append(self, item: PackageToTest) -> None:
        self.data.append(item)


with open("pypi.json", "r") as f:
    packages = Packages(data=json.load(f)["data"])

packages.data.sort(key=lambda p: p.stars or 0, reverse=True)

# Use to test the below code for now
# packages = Packages(data=[PackageToTest(name="sqlalchemy", versions=["1.4.10"])])

failed_packages = []
for i, package in enumerate(packages.data):
    # if i < 5:
    #     continue
    try:
        use(package.name, version=package.versions[-1], modes=use.auto_install)
    except RuntimeWarning as e:
        if str(e).startswith("Failed to auto-install "):
            hashes = re.findall("hashes={([a-z0-9A-Z', ]+)}", str(e))[0]
            hashes = {_hash.strip("'") for _hash in hashes.split(", ")}
    try:
        module = use(package.name, version=package.versions[-1], modes=use.auto_install, hashes=hashes)
        assert module

    except Exception as e:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        failed_packages.append(
            {
                "name": package.name,
                "version": package.versions[-1],
                "stars": package.stars,
                "err": {"type": str(exc_type), "value": str(exc_value), "traceback": tb.split("\n")},
                "retry": f"""use('{package.name}', version='{package.versions[-1]}', modes=use.auto_install, hashes={hashes})""",
            }
        )

    if i > 10:
        break

print(json.dumps(failed_packages))
