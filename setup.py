import os
import sys

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
src = os.path.join(here, "src/use")

# Instead of doing the obvious thing (importing 'use' directly and just reading '__version__'),
# we are parsing the version out of the source AST here, because if the user is missing any
# dependencies at setup time, an import error would prevent the installation.
# Two simple ways to verify the installation using this setup.py file:
#
#     python3 setup.py develop
#
#  or:
#
#    python3 setup.py install
#
import ast

with open(os.path.join(src, "use.py")) as f:
    mod = ast.parse(f.read())
    version = (
        [
            *filter(
                lambda n: isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "__version__",
                mod.body,
            )
        ][0]
        .value.values[1]
        .value
    )

meta = {
    "name": "justuse",
    "description": "A beautiful, simple and explicit way to import modules in Python with advanced features.",
    "license": "MIT",
    "url": "https://github.com/amogorkon/justuse",
    "version": version,
    "author": "Anselm Kiefner",
    "author_email": "justuse-pypi@anselm.kiefner.de",
    "python_requires": ">=3.8",
    "keywords": [
        "installing",
        "packages",
        "hot reload",
        "auto install",
        "aspect oriented",
        "version checking",
        "functional",
    ],
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
    ],
}

requires = (
    "requests(>= 2.24.0)",
    "packaging(>= 21.0)",
    "typeguard(>= 2.12.1)",
    "pip(>= 21.2.1)",
    "furl(>= 2.1.2)",
    "wheel(>= 0.36.2)",
    "icontract(>= 2.5.4)",
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
    zip_safe=True,
    **meta
)
