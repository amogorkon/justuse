import io
import sys
import json
from typing import List, Optional
import re
import logging
import traceback
import contextlib

import use
from pydantic import BaseModel


def start_capture_logs():
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    logging.root.addHandler(ch)
    return log_capture_string


def get_capture_logs(log_capture_string):
    log_contents = log_capture_string.getvalue()
    log_capture_string.close()
    return log_contents


class PackageToTest(BaseModel):
    name: str
    versions: List[str]
    repo: Optional[str] = None
    stars: Optional[int] = None

    @property
    def safe_versions(self):
        return list(filter(lambda k: k.replace(".", "").isnumeric(), self.versions))


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
passing_packages = []
for i, package in enumerate(packages.data):
    # if i < 5:
    #     continue

    try:
        use(package.name, version=package.safe_versions[-1], modes=use.auto_install)
    except RuntimeWarning as e:
        if str(e).startswith("Failed to auto-install "):
            hashes = re.findall("hashes={([a-z0-9A-Z', ]+)}", str(e))[0]
            hashes = {_hash.strip("'") for _hash in hashes.split(", ")}

    logs = start_capture_logs()
    try:
        module = use(package.name, version=package.safe_versions[-1], modes=use.auto_install, hashes=hashes)
        assert module
        passing_packages.append(
            {
                "name": package.name,
                "version": package.safe_versions[-1],
                "stars": package.stars,
                "retry": f"""use('{package.name}', version='{package.safe_versions[-1]}', modes=use.auto_install, hashes={hashes})""",
            }
        )

    except Exception as e:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        failed_packages.append(
            {
                "name": package.name,
                "version": package.safe_versions[-1],
                "stars": package.stars,
                "err": {
                    "type": str(exc_type),
                    "value": str(exc_value),
                    "traceback": tb.split("\n"),
                    "logs": get_capture_logs(logs).split("\n"),
                },
                "retry": f"""use('{package.name}', version='{package.safe_versions[-1]}', modes=use.auto_install, hashes={hashes})""",
            }
        )

        if i > 10:
            break

with open("FAIL.json", "w") as f:
    json.dump(failed_packages, f, indent=2, sort_keys=True)

with open("PASS.json", "w") as f:
    json.dump(passing_packages, f, indent=2, sort_keys=True)
