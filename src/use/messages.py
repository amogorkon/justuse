"""
Collection of the messages directed to the user.
How it works is quite magical - the lambdas prevent the f-strings from being prematuraly evaluated, and are only evaluated once returned.
Fun fact: f-strings are firmly rooted in the AST.
"""

import webbrowser
from collections import defaultdict, namedtuple
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from shutil import copy
from statistics import geometric_mean, median, stdev

from beartype import beartype
from jinja2 import Environment, FileSystemLoader, select_autoescape

import use
from use import __version__, config, home
from use.hash_alphabet import hexdigest_as_JACK
from use.pydantics import PyPI_Release, Version

env = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape()
)


def std(times):
    return stdev(times) if len(times) > 1 else 0


def _web_tinny_profiler(timings):
    copy(Path(__file__).absolute().parent / r"templates/profiling.css", home / "profiling.css")
    DAT = namedtuple("DECORATOR", "qualname name min geom_mean median stdev len total")
    timings_ = [
        DAT(
            getattr(
                func, "__qualname__", getattr(func, "__module__", "") + getattr(func, "__name__", str(func))
            ),
            func.__name__,
            min(times),
            geometric_mean(times),
            median(times),
            std(times),
            len(times),
            sum(times),
        )
        for func, times in timings.items()
    ]
    with open(home / "profiling.html", "w", encoding="utf-8") as file:
        args = {
            "timings": timings_,
        }
        file.write(env.get_template("profiling.html").render(**args))
    webbrowser.open(f"file://{home}/profiling.html")


def _web_aspectized(decorators, functions):
    copy(Path(__file__).absolute().parent / r"templates/aspects.css", home / "aspects.css")
    DAT = namedtuple("DECORATOR", "name func")
    redecorated = []
    for ID, funcs in decorators.items():
        name = functions[ID][-1].__name__
        redecorated.append(DAT(name, funcs))

    with open(home / "aspects.html", "w", encoding="utf-8") as file:
        args = {
            "decorators": redecorated,
            "functions": functions,
        }
        file.write(env.get_template("aspects.html").render(**args))
    webbrowser.open(f"file://{home}/aspects.html")


@beartype
def _web_aspectized_dry_run(
    *, decorator: Callable, hits: list, check: Callable, pattern: str, module_name: str
):
    copy(Path(__file__).absolute().parent / r"templates/aspects.css", home / "aspects.css")
    with open(home / "aspects_dry_run.html", "w", encoding="utf-8") as file:
        args = {
            "decorator": decorator,
            "hits": hits,
            "check": check,
            "module_name": module_name,
        }
        file.write(env.get_template("aspects_dry_run.html").render(**args))
    webbrowser.open(f"file://{home}/aspects_dry_run.html")


def _web_pebkac_no_version_no_hash(*, name, package_name, version, no_browser: bool):
    if not no_browser:
        webbrowser.open(f"https://snyk.io/advisor/python/{package_name}")
    return f"""Please specify version and hash for auto-installation of {package_name!r}.
{"" if no_browser else "A webbrowser should open to the Snyk Advisor to check whether the package is vulnerable or malicious."}
If you want to auto-install the latest version, try the following line to select all viable hashes:
use("{name}", version="{version!s}", modes=use.auto_install)"""


@beartype
def _web_pebkac_no_hash(
    *,
    name: str,
    package_name: str,
    version: Version,
    releases: list[PyPI_Release],
):
    copy(Path(__file__).absolute().parent / r"templates/stylesheet.css", home / "stylesheet.css")
    entry = namedtuple("Entry", "python platform hash_name hash_value jack_value")
    table = defaultdict(lambda: [])
    for rel in (rel for rel in releases if rel.version == version):
        for hash_name, hash_value in rel.digests.items():
            if hash_name not in (x.name for x in use.Hash):
                continue
            table[hash_name].append(
                entry(
                    rel.python_tag,
                    rel.platform_tag,
                    hash_name,
                    hash_value,
                    hexdigest_as_JACK(hash_value),
                )
            )

    with open(home / "web_exception.html", "w", encoding="utf-8") as file:
        args = {
            "name": name,
            "package_name": package_name,
            "version": version,
            "table": table,
        }
        file.write(env.get_template("hash-presentation.html").render(**args))

    # from base64 import b64encode
    # def data_uri_from_html(html_string):
    #    return f'data:text/html;base64,{b64encode(html_string.encode()).decode()}'
    webbrowser.open(f"file://{home}/web_exception.html")


class UserMessage(Enum):
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
    pebkac_missing_hash = (
        lambda *, name, package_name, version, recommended_hash, no_browser: f"""Failed to auto-install {package_name!r} because hashes aren't specified.
        {"" if no_browser else "A webbrowser should open with a list of available hashes for different platforms for you to pick."}"
        If you want to use the package only on this platform, this should work:
    use("{name}", version="{version!s}", hashes={recommended_hash!r}, modes=use.auto_install)"""
    )
    pebkac_unsupported = (
        lambda package_name: f"We could not find any version or release for {package_name} that could satisfy our requirements!"
    )
    pip_json_mess = (
        lambda package_name, target_version: f"Tried to auto-install {package_name} {target_version} but failed because there was a problem with the JSON from PyPI."
    )
    pebkac_no_version_no_hash = _web_pebkac_no_version_no_hash
    cant_import = lambda name: f"No pkg installed named {name} and auto-installation not requested. Aborting."
    cant_import_no_version = (
        lambda package_name: f"Failed to auto-install '{package_name}' because no version was specified."
    )

    no_distribution_found = (
        lambda package_name, version, last_version: f"Failed to find any distribution for {package_name} version {version} that can be run on this platform. (For your information, the most recent version of {package_name} is {last_version})"
    )

    no_recommendation = (
        lambda package_name, version: f"We could not find any release for {package_name} {version} that appears to be compatible with this platform. Check your browser for a list of hashes and select manually."
    )


class StrMessage(UserMessage):
    cant_import = (
        lambda package_name: f"No pkg installed named {package_name} and auto-installation not requested. Aborting."
    )


class TupleMessage(UserMessage):
    pass


class KwargMessage(UserMessage):
    pass


def _web_aspectizing_overview(*, decorator, check, pattern, visited, hits):
    msg = f"""
<html>
<head>
<script src="/brython.js"></script>
</head>
<body>
<ul>
    {''.join(f"<li>{h}</li>" for h in hits)}
</ul>
</body>
</html>
"""

    with open(use.home / "aspectizing_overview.html", "w") as f:
        f.write(msg)
    if not config.testing:
        webbrowser.open(use.home / "aspectizing_overview.html")
    return msg
