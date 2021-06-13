
import importlib
import importlib
import os
import sys

from setuptools import find_packages
from setuptools import setup

here = (os.path.abspath(os.path.dirname(__file__)))
src = os.path.join(here, "src/use")
sys.path.insert(0, src)

import use

print(2323, use.__version__)

meta={
    "name":"justuse",
    "description":"A self-documenting, functional way to import modules in Python with advanced features.",
    "license":"MIT",
    "url":"https://github.com/amogorkon/justuse",
    "version": use.__version__,
    "author":"Anselm Kiefner",
    "author_email":"justuse-pypi@anselm.kiefner.de",
    "python_requires":">=3.8",
    "keywords":["import","reload"],
    "classifiers":[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
    ]
}


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
  
setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_name="use",
    install_requires=[
            "anyio >= 3.1.0",
            "mmh3 >= 2.2.0",
            "requests >= 2.24.0",
            "yarl >= 1.6.3",
    ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    zip_safe=False,
    **meta
)
