import json
from os import PathLike
import subprocess
import shutil
from pathlib import Path

import use
from test_single import Packages


def manage_disk(max_size=5_000_000_000):
    if not (use.home / "venv").exists():
        return
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


def clear_cache():
    results_dir = Path("results")
    if results_dir.exists():
        shutil.rmtree("results")


def run_test(packages: Packages, results_dir: PathLike, max_to_run: int = 1, max_venv_space: int = 5_000_000_000):
    for i, package in enumerate(packages.data):
        if i >= max_to_run:
            break
        if package.name in NAUGHTY_PACKAGES:
            continue

        manage_disk(max_size=max_venv_space)

        subprocess.call(f"python test_single.py {i}", shell=True)
        n_passed = len(list((results_dir / "pass").glob("*.json")))
        n_failed = len(list((results_dir / "fail").glob("*.json")))
        print(i, package.name, n_failed + n_passed, n_failed, n_passed, f"{100 * n_passed / (n_failed + n_passed)}%")


def combine_package_output(results_dir: PathLike, folder: str):

    packages = []
    for file_path in (results_dir / folder).glob("*.json"):
        with open(file_path, "r") as f:
            packages.append(json.load(f))

    with open(f"{folder}.json", "w") as f:
        json.dump(packages, f, indent=2, sort_keys=True)

    return packages


if __name__ == "__main__":
    NAUGHTY_PACKAGES = ["assimp", "metakernel", "pscript", "airflow"]

    with open("pypi.json", "r") as f:
        packages = Packages(data=json.load(f)["data"])

    packages.data.sort(key=lambda p: p.stars or 0, reverse=True)

    results_dir = Path("results")

    clear_cache()
    run_test(packages, results_dir, 100, max_venv_space=50_000_000_000)
    passed = combine_package_output(results_dir, "pass")
    failed = combine_package_output(results_dir, "fail")
    clear_cache()

    print("Total: ", len(packages.data))
    print("Failed: ", len(failed))
    print("Passed: ", len(passed))
    print("Pass rate: ", 100 * (len(passed) / len(failed + passed)))
