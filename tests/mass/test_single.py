import io
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Tuple
import re
import logging
import traceback
import packaging

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


def test_package(package: PackageToTest) -> Tuple[bool, Dict]:
    use_version = package.versions[-1]
    if len(package.safe_versions) != 0:
        use_version = package.safe_versions[-1]

    log1 = start_capture_logs()
    try:
        use(package.name, version=use_version, modes=use.auto_install)
    except RuntimeWarning as e:
        if str(e).startswith("Failed to auto-install "):
            hashes = re.findall("hashes={([a-z0-9A-Z', ]+)}", str(e))[0]
            hashes = {_hash.strip("'") for _hash in hashes.split(", ")}
        else:
            exc_type, exc_value, _ = sys.exc_info()
            tb = traceback.format_exc()
            return (
                False,
                {
                    "name": package.name,
                    "version": use_version,
                    "stars": package.stars,
                    "err": {
                        "type": str(exc_type),
                        "value": str(exc_value),
                        "traceback": tb.split("\n"),
                        "logs": get_capture_logs(log1).split("\n"),
                    },
                },
            )

    except packaging.version.InvalidVersion:
        return (
            False,
            {
                "name": package.name,
                "stars": package.stars,
                "err": {"type": "InvalidVersion", "value": package.versions, "picked": use_version},
            },
        )

    except Exception as e:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        return (
            False,
            {
                "name": package.name,
                "version": use_version,
                "stars": package.stars,
                "err": {
                    "type": str(exc_type),
                    "value": str(exc_value),
                    "traceback": tb.split("\n"),
                    "logs": get_capture_logs(log1).split("\n"),
                },
            },
        )

    else:
        get_capture_logs(log1)

    logs = start_capture_logs()
    try:
        module = use(package.name, version=use_version, modes=use.auto_install, hashes=hashes)
        assert module
        return (
            True,
            {
                "name": package.name,
                "version": use_version,
                "stars": package.stars,
                "retry": f"""use('{package.name}', version='{use_version}', modes=use.auto_install, hashes={hashes})""",
            },
        )

    except Exception as e:
        exc_type, exc_value, _ = sys.exc_info()
        tb = traceback.format_exc()
        return (
            False,
            {
                "name": package.name,
                "version": use_version,
                "stars": package.stars,
                "err": {
                    "type": str(exc_type),
                    "value": str(exc_value),
                    "traceback": tb.split("\n"),
                    "logs": get_capture_logs(logs).split("\n"),
                },
                "retry": f"""use('{package.name}', version='{use_version}', modes=use.auto_install, hashes={hashes})""",
            },
        )


if __name__ == "__main__":
    fail_dir = Path("results") / "fail"
    pass_dir = Path("results") / "pass"

    fail_dir.mkdir(parents=True, exist_ok=True)
    pass_dir.mkdir(parents=True, exist_ok=True)

    with open("pypi.json", "r") as f:
        packages = Packages(data=json.load(f)["data"])

    index = sys.argv[1]
    if index.isdigit():
        index = int(index)
        package = packages.data[index]
    else:
        package = next((item for item in packages.data if item.name == index), None)

    did_work, info = test_package(package)
    out_dir = pass_dir if did_work else fail_dir

    with open(out_dir / f"{package.name}.json", "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)

