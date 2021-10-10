import io
import sys
import json
from typing import List, Optional
import re
import logging
import traceback
import contextlib
import subprocess
import shutil
import packaging
from copy import deepcopy


import use
from pydantic import BaseModel


def start_capture_logs():
    log_capture_string = io.StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)
    logging.root.handlers = [ch]
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
NAUGHTY_PACKAGES = ["assimp", "metakernel", "pscript"]

# 636/1321


def manage_disk(max_size=5_000_000_000):
    if not any((use.home / "venv").iterdir()):
        return
    current_usage = int(subprocess.check_output(["du", "-sb", f"{use.home}/venv"]).split(b"\t")[0])
    if current_usage > max_size:
        process = subprocess.Popen(f"du -sb {use.home}/venv/* | sort -n -r", shell=True, stdout=subprocess.PIPE)
        venv_usages = process.communicate()[0].split(b"\n")
        for venv in venv_usages:
            try:
                size, path = venv.split(b"\t")
                path = path.decode()
                size = int(size)
                venv_package = path.split("/")[-1]

                print(f"Deleting {venv_package} to make extra space, freed {size/1_000_000} MB")
                shutil.rmtree(path)
                current_usage -= size
                if current_usage < max_size:
                    break
            except:
                continue


failed_packages = []
passing_packages = []
old_modules = deepcopy(list(sys.modules.keys()))
for i, package in enumerate(packages.data):
    new_imports = [k for k in sys.modules.keys() if k not in old_modules]
    for new_import in new_imports:
        del sys.modules[new_import]
    if package.name in NAUGHTY_PACKAGES:
        continue

    manage_disk(max_size=1_000)
    print(i, package.name, len(failed_packages), len(passing_packages))
    if len(package.safe_versions) == 0:
        failed_packages.append(
            {
                "name": package.name,
                "stars": package.stars,
                "err": {"type": "NoSafeVersionsError", "value": package.versions,},
            }
        )
        continue
    log1 = start_capture_logs()
    try:
        use(package.name, version=package.safe_versions[-1], modes=use.auto_install)
    except RuntimeWarning as e:
        if str(e).startswith("Failed to auto-install "):
            hashes = re.findall("hashes={([a-z0-9A-Z', ]+)}", str(e))[0]
            hashes = {_hash.strip("'") for _hash in hashes.split(", ")}
        else:
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
                        "logs": get_capture_logs(log1).split("\n"),
                    },
                }
            )
            continue
    except packaging.version.InvalidVersion:
        failed_packages.append(
            {
                "name": package.name,
                "stars": package.stars,
                "err": {"type": "InvalidVersion", "value": package.versions, "picked": package.safe_versions[-1]},
            }
        )
        continue
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
                    "logs": get_capture_logs(log1).split("\n"),
                },
            }
        )
        continue
    else:
        get_capture_logs(log1)

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
        continue

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
        continue


with open("FAIL.json", "w") as f:
    json.dump(failed_packages, f, indent=2, sort_keys=True)

with open("PASS.json", "w") as f:
    json.dump(passing_packages, f, indent=2, sort_keys=True)

print(len(packages.data))
print("Failed: ", len(failed_packages))
print("Passed: ", len(passing_packages))
