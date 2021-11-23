"""
Collection of the messages directed to the user.
How it works is quite magical - the lambdas prevent the f-strings from being prematuraly evaluated, and are only evaluated once returned.
Fun fact: f-strings are firmly rooted in the AST.
"""

import webbrowser
from enum import Enum

import use
from use import __version__, config
from use.pypi_model import Version


def _web_no_version_or_hash_provided(*, name, package_name, version, hashes):
    if not config["testing"]:
        webbrowser.open(f"https://snyk.io/advisor/python/{package_name}")
    return f"""Please specify version and hash for auto-installation of {package_name!r}.
A webbrowser will open to the Snyk Advisor to check whether the package is vulnerable.
If you want to auto-install the latest version:
use("{name}", version="{version!s}", hashes={hashes!r}, modes=use.auto_install)"""


def _web_pebkac_missing_hash(name, package_name, version, hashes):
    if not config["testing"]:
        with open(use.home / "web_exception.html", "w") as f:
            f.write(
                f"""<html><body>
    <h1>{package_name!r}</h1>
    {config}
    <p>
    Please specify the hash for auto-installation.
    </p>
    <p>
    </p>
    </body></html>"""
            )
        webbrowser.open(f"file://{use.home / 'web_exception.html'}")
    return f"""Failed to auto-install {package_name!r} because hashes aren't specified.
        A webbrowser will open with a list of available hashes for different platforms.
        If you only want to use the package on this platform, this may work:
    use("{name}", version="{version!s}", hashes={hashes!r}, modes=use.auto_install)"""


class Message(Enum):
    not_reloadable = (
        lambda name: f"Beware {name} also contains non-function objects, it may not be safe to reload!"
    )
    couldnt_connect_to_db = (
        lambda e: f"Could not connect to the registry database, please make sure it is accessible. ({e})"
    )
    use_version_warning = (
        lambda max_version: f"""Justuse is version {Version(__version__)}, but there is a newer version {max_version} available on PyPI.
To find out more about the changes check out https://github.com/amogorkon/justuse/wiki/What's-new
Please consider upgrading via
python -m pip install -U justuse
"""
    )
    cant_use = (
        lambda thing: f"Only pathlib.Path, yarl.URL and str are valid sources of things to import, but got {type(thing)}."
    )
    web_error = (
        lambda url, response: f"Could not load {url} from the interwebs, got a {response.status_code} error."
    )
    no_validation = (
        lambda url, hash_algo, this_hash: f"""Attempting to import from the interwebs with no validation whatsoever!
To safely reproduce:
use(use.URL('{url}'), hash_algo=use.{hash_algo}, hash_value='{this_hash}')"""
    )
    version_warning = (
        lambda package_name, target_version, this_version: f"{package_name} expected to be version {target_version}, but got {this_version} instead"
    )
    ambiguous_name_warning = (
        lambda package_name: f"Attempting to load the pkg '{package_name}', if you rather want to use the local module: use(use._ensure_path('{package_name}.py'))"
    )
    no_version_provided = (
        lambda: "No version was provided, even though auto_install was specified! Trying to load classically installed pkg instead."
    )
    classically_imported = (
        lambda name, this_version: f'Classically imported \'{name}\'. To pin this version: use("{name}", version="{this_version}")'
    )
    pebkac_missing_hash = _web_pebkac_missing_hash
    pebkac_unsupported = (
        lambda package_name: f"We could not find any version or release for {package_name} that could satisfy our requirements!"
    )
    pip_json_mess = (
        lambda package_name, target_version: f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI."
    )
    no_version_or_hash_provided = _web_no_version_or_hash_provided
    cant_import = lambda name: f"No pkg installed named {name} and auto-installation not requested. Aborting."
    cant_import_no_version = (
        lambda package_name: f"Failed to auto-install '{package_name}' because no version was specified."
    )
    venv_unavailable = (
        lambda python_exe, python_version, python_platform: f"""
Your system does not have a usable 'venv' pkg for this version of Python:
   Path =     {python_exe}
   Version =  {python_version}
   Platform = {python_platform}

Please run:
   sudo apt update
   sudo apt install python3-venv
to install the necessary packages.

You can test if your version of venv is working by running:
  {python_exe} -m venv testenv && ls -lA testenv/bin/python
"""
    )
    no_distribution_found = (
        lambda package_name, version: f"Failed to find any distribution for {package_name} with version {version} that can be run this platform!"
    )


class StrMessage(Message):
    cant_import = (
        lambda package_name: f"No pkg installed named {package_name} and auto-installation not requested. Aborting."
    )


class TupleMessage(Message):
    pass


class KwargsMessage(Message):
    pass
