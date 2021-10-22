import json
import os
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]


def get_soup(page: int):
    url = "https://anaconda.org/anaconda/repo?sort=time.modified&access=public&sort_order=desc&type=all&page={page}"
    return BeautifulSoup(requests.get(url.format(page=page)).text, features="html.parser")


def parse_package(soup):
    return {
        "name": soup.select_one("span.packageName").text,
        "href": soup.select_one("a[data-package]")["href"],
    }


def parse_packages(soup):
    return [parse_package(s) for s in soup.select("td.pkg-title")]


def find_all_package_names():

    home = get_soup(page=1)
    number_of_pages = int(home.select_one("ul.pagination li[aria-disabled] a").text.split()[-1])

    packages = parse_packages(home)
    for page in range(2, number_of_pages + 1):
        soup = get_soup(page=page)
        packages += parse_packages(soup)

    return packages


def optional_text(soup, default=""):
    if soup is None:
        return default
    return soup.text


def find_meta(pkg: Dict[str, str]):
    url = f"https://pypi.org/pypi/{pkg['name']}/json"
    r = requests.get(url)
    if r.status_code != 200:
        return
    meta = r.json()
    link_options = [meta["info"]["home_page"]]
    if (project_urls := meta["info"].get("project_urls")) is not None:
        link_options += list(project_urls.values())
    owner, repo, url = get_github(link_options)
    stars = get_stars(owner, repo)
    base = {"name": pkg["name"], "versions": [version for version in meta["releases"].keys()]}
    if stars < 0:
        return base
    return {**base, **{"stars": stars, "repo": url}}


def get_github(urls: List[str]):
    for url in urls:
        if not isinstance(url, str):
            continue
        url = url.strip().strip("/")
        base1 = "https://github.com/"
        base2 = "http://github.com/"
        paths = []
        if url.startswith(base1):
            paths = url[len(base1) :].split("/")
            base = base1
        if url.startswith(base2):
            paths = url[len(base2) :].split("/")
            base = base2

        if len(paths) >= 2:
            return paths[0], paths[1], base + paths[0] + "/" + paths[1]

    return None, None, None


def get_stars(owner: str, repo: str):
    if owner is None or repo is None:
        return -1

    query = f"""query {{
        repository(owner: "{owner}", name: "{repo}") {{
            stargazers {{
            totalCount
            }}
        }}
    }}"""
    r = requests.post(
        "https://api.github.com/graphql",
        json={"query": query},
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
    )
    if r.status_code == 200:
        data = r.json()
        try:
            if data is None:
                print("Hit GitHub rate limit, sleeping")
                time.sleep(60)
                return get_stars(owner, repo)
            return data["data"]["repository"]["stargazers"]["totalCount"]
        except:
            print(data)
            return -1

    print(r.status_code)
    raise Exception(r.text)


def try_to_get_github_stars(pkg):
    owner, repo = get_github([pkg["urls"]["dev"], pkg["urls"]["home"]])
    if owner is None:
        return -1
    return get_stars(owner, repo)


def main():

    ## Step 1 - get all conda pkg names and dump to file
    # with open("tmp.json", "r") as f:
    #     packages = json.load(f)
    packages = find_all_package_names()

    ## Step 2 - go to pypi and find metadata (try to get stars from github)
    pypi_packages = Packages()
    for i, pkg in enumerate(packages):
        meta = find_meta(pkg)
        if meta is None:
            print("Not on Pypi", pkg)
            continue
        pypi_packages.append(PackageToTest(**meta))
        print(i)

    ## Step 4 Dump out
    with open("pypi.json", "w") as f:
        json.dump(pypi_packages.dict(), f, indent=2, sort_keys=True)


class PackageToTest(BaseModel):
    name: str
    versions: List[str]
    repo: Optional[str] = None
    stars: Optional[int] = None


class Packages(BaseModel):
    data: List[PackageToTest] = []

    def append(self, item: PackageToTest) -> None:
        self.data.append(item)


if __name__ == "__main__":
    main()
