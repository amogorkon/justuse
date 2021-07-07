
import os
import sys

from setuptools import find_packages
from setuptools import setup

here = (os.path.abspath(os.path.dirname(__file__)))
src = os.path.join(here, "src/use")
sys.path.insert(0, src)

import ast
with open(os.path.join(src, "use.py")) as f:
    mod = ast.parse(f.read())
    version = [*filter(lambda n: isinstance(n, ast.Assign) \
                             and isinstance(n.targets[0], ast.Name) \
                             and n.targets[0].id == "__version__", mod.body)][0].value.value

meta={
    "name":"justuse",
    "description":"A beautiful, simple and explicit way to import modules in Python with advanced features.",
    "license":"MIT",
    "url":"https://github.com/amogorkon/justuse",
    "version": version,
    "author":"Anselm Kiefner",
    "author_email":"justuse-pypi@anselm.kiefner.de",
    "python_requires":">=3.8",
    "keywords":["installing", "packages", "hot reload", "auto install", "aspect oriented", "version checking", "functional"],
    "classifiers":[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
    ]
}

requires=(
    "anyio(>=3.1.0)",
    "mmh3(>= 2.2.0)",
    "requests(>= 2.24.0)",
    "yarl(>= 1.6.3)",
    "packaging(>= 1.0.0)"
)


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
  
setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_name="use",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    requires=requires,
    install_requires=requires,
    zip_safe=False,
    **meta
)
