import ast
import inspect
import re
import sys
import re
from functools import lru_cache as cache
from itertools import takewhile
from pathlib import Path
from textwrap import dedent
from typing import Dict, FrozenSet, List, Optional

import packaging
from icontract import require
from packaging import tags
from packaging.specifiers import SpecifierSet
from packaging.version import Version as PkgVersion
from pip._internal.utils import compatibility_tags
from pydantic import BaseModel


class Version(PkgVersion):
    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Version):
            return args[0]
        else:
            return super(cls, Version).__new__(cls)

    def __init__(self, versionstr=None, *, major=0, minor=0, patch=0):
        if isinstance(versionstr, Version):
            return
        if not (versionstr or major or minor or patch):
            raise ValueError("Version must be initialized with either a string or major, minor and patch")
        if major or minor or patch:
            # string as only argument, no way to construct a Version otherwise - WTF
            return super().__init__(".".join((str(major), str(minor), str(patch))))
        return super().__init__(versionstr)

    def __iter__(self):
        yield from self.release

    def __repr__(self):
        return (
            'use.Version("'
            + ".".join(
                map(
                    str,
                    (
                        *self.release[0:-1],
                        str(self.release[-1]) + (self.pre or ("", ""))[0] + str((self.pre or ("", ""))[1]),
                    ),
                )
            )
            + '")'
        )

    def __hash__(self):
        return hash(self._version)


class PlatformTag:
    def __init__(self, platform: str):
        self.platform = platform

    def __str__(self):
        return self.platform

    def __repr__(self):
        return f"use.PlatformTag({self.platform!r})"

    def __hash__(self):
        return hash(self.platform)

    @require(lambda self, other: isinstance(other, self.__class__))
    def __eq__(self, other):
        return self.platform == other.platform


class JustUse_Info(BaseModel):
    distribution: Optional[str]
    version: Optional[str]
    build_tag: Optional[str]
    python_tag: Optional[str] = ""
    abi_tag: Optional[str]
    platform_tag: Optional[str] = "any"
    ext: Optional[str]


# class PyPI_Downloads(BaseModel):
#     last_day: int
#     last_month: int
#     last_week: int


class PyPI_Info(BaseModel):
    # author: Optional[str]
    # author_email: Optional[str]
    # bugtrack_url: Optional[str]
    # classifiers: List[str]
    # description: Optional[str]
    # description_content_type: Optional[str]
    # docs_url: Optional[str]
    download_url: Optional[str]
    # downloads: PyPI_Downloads
    # home_page: Optional[str]
    # keywords: Optional[str]
    # license: Optional[str]
    # maintainer: Optional[str]
    # maintainer_email: Optional[str]
    name: str
    package_url: str
    platform: Optional[str]
    # project_url: str
    # project_urls: Optional[Dict[str, str]]
    # release_url: str
    requires_dist: Optional[List[str]]
    requires_python: Optional[str] = ""
    # summary: Optional[str]
    version: Optional[str]
    yanked: bool
    yanked_reason: Optional[str]


class PyPI_Digests(BaseModel):
    md5: str
    sha256: str


class PyPI_Release(BaseModel):
    # comment_text: Optional[str]
    digests: PyPI_Digests
    # downloads: int
    filename: str
    has_sig: bool
    md5_digest: str
    packagetype: Optional[str]
    python_version: str
    requires_python: Optional[str] = ""
    # size: int
    # upload_time: str
    # upload_time_iso_8601: str
    url: str
    yanked: bool
    yanked_reason: Optional[str]

    version: Optional[Version]

    class Config:
        arbitrary_types_allowed = True

    @property
    def justuse(self) -> JustUse_Info:
        pp = Path(self.filename)
        if ".tar" in self.filename:
            ext = self.filename[self.filename.index(".tar") + 1 :]
        else:
            ext = pp.name[len(pp.stem) + 1 :]
        rest = pp.name[0 : -len(ext) - 1]

        not_dash = lambda name: f"(?P<{name}>[^-]+)"
        not_dash_with_int = lambda name: f"(?P<{name}>[0-9][^-]*)"
        match = re.match(
            f"{not_dash('distribution')}-{not_dash('version')}-?{not_dash_with_int('build_tag')}?-?{not_dash('python_tag')}?-?{not_dash('abi_tag')}?-?{not_dash('platform_tag')}?",
            rest,
        )
        if match:
            return JustUse_Info(**_delete_none(match.groupdict()), ext=ext)
        return JustUse_Info()


class PyPI_Project(BaseModel):
    info: PyPI_Info
    last_serial: int
    releases: Dict[str, List[PyPI_Release]]
    urls: List[PyPI_Release]

    def __init__(self, **kwargs):
        kwargs["releases"] = {k: [{**_v, "version": Version(k)} for _v in v] for k, v in kwargs["releases"].items()}
        super().__init__(**kwargs)

    def filter_by_version(self, version: str) -> "PyPI_Project":
        return PyPI_Project(
            **{
                **self.dict(),
                **{
                    "releases": {version: [v.dict() for v in self.releases[version]]}
                    if self.releases.get(version)
                    else {}
                },
            }
        )

    def filter_by_platform(self, tags: FrozenSet[PlatformTag], sys_version: Version) -> "PyPI_Project":
        filtered = {}
        for ver, releases in self.releases.items():
            filtered[ver] = list(
                (
                    rel.dict()
                    for rel in releases
                    if _is_compatible(rel, sys_version=sys_version, platform_tags=tags, include_sdist=True)
                )
            )
        return PyPI_Project(**{**self.dict(), **{"releases": filtered}})

    def filter_by_version_and_current_platform(self, version: str) -> "PyPI_Project":
        return self.filter_by_version(version).filter_by_platform(tags=get_supported(), sys_version=_sys_version())


def _delete_none(a_dict: Dict[str, object]) -> Dict[str, object]:
    for k, v in tuple(a_dict.items()):
        if v is None or v == "":
            del a_dict[k]
    return a_dict


def _sys_version():
    return Version(".".join(map(str, sys.version_info[0:3])))


@cache
def get_supported() -> FrozenSet[PlatformTag]:
    """
    Results of this function are cached. They are expensive to
    compute, thanks to some heavyweight usual players
    (*ahem* pip, package_resources, packaging.tags *cough*)
    whose modules are notoriously resource-hungry.

    Returns a set containing all platform _platform_tags
    supported on the current system.
    """
    items: List[PlatformTag] = []

    for tag in compatibility_tags.get_supported():
        items.append(PlatformTag(platform=tag.platform))
    for tag in packaging.tags._platform_tags():
        items.append(PlatformTag(platform=str(tag)))

    return frozenset(items)


@cache
def _is_version_satisfied(specifier: str, sys_version) -> bool:
    """
    SpecifierSet("") matches anything, no need to artificially
    lock down versions at this point

    @see https://warehouse.readthedocs.io/api-reference/json.html
    @see https://packaging.pypa.io/en/latest/specifiers.html
    """
    specifiers = SpecifierSet(specifier or "")
    is_match = sys_version in specifiers
    return is_match


def _is_compatible(info: PyPI_Release, sys_version, platform_tags, include_sdist=None) -> bool:
    """Return true if the artifact described by 'info'
    is compatible with the current or specified system."""
    specifier = info.requires_python

    return (
        ((not specifier or _is_version_satisfied(specifier, sys_version)))
        and _is_platform_compatible(info, platform_tags, include_sdist)
        and not info.yanked
        and (include_sdist or info.justuse.ext not in ("tar", "tar.gz" "zip"))
    )


# Since we have quite a bit of functional code that black would turn into a sort of arrow antipattern with lots of ((())),
# we use @pipes to basically enable polish notation which allows us to avoid most parentheses.
# source >> func(args) is equivalent to func(source, args) and
# source << func(args) is equivalent to func(args, source), which can be chained arbitrarily.
# Rules:
# 1) apply pipes only to 3 or more nested function calls
# 2) no pipes on single lines, since mixing << and >> is just confusing (also, having pipes on different lines has other benefits beside better readability)
# 3) don't mix pipes with regular parenthesized function calls, that's just confusing
# See https://github.com/robinhilliard/pipes/blob/master/pipeop/__init__.py for details and credit.
class _PipeTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        if not isinstance(node.op, (ast.LShift, ast.RShift)):
            return node
        if not isinstance(node.right, ast.Call):
            return self.visit(
                ast.Call(
                    func=node.right,
                    args=[node.left],
                    keywords=[],
                    starargs=None,
                    kwargs=None,
                    lineno=node.right.lineno,
                    col_offset=node.right.col_offset,
                )
            )
        node.right.args.insert(0 if isinstance(node.op, ast.RShift) else len(node.right.args), node.left)
        return self.visit(node.right)


def pipes(func_or_class):
    if inspect.isclass(func_or_class):
        decorator_frame = inspect.stack()[1]
        ctx = decorator_frame[0].f_locals
        first_line_number = decorator_frame[2]
    else:
        ctx = func_or_class.__globals__
        first_line_number = func_or_class.__code__.co_firstlineno
    source = inspect.getsource(func_or_class)
    tree = ast.parse(dedent(source))
    ast.increment_lineno(tree, first_line_number - 1)
    source_indent = sum(1 for _ in takewhile(str.isspace, source)) + 1
    for node in ast.walk(tree):
        if hasattr(node, "col_offset"):
            node.col_offset += source_indent
    tree.body[0].decorator_list = [
        d
        for d in tree.body[0].decorator_list
        if isinstance(d, ast.Call) and d.func.id != "pipes" or isinstance(d, ast.Name) and d.id != "pipes"
    ]
    tree = _PipeTransformer().visit(tree)
    code = compile(tree, filename=(ctx["__file__"] if "__file__" in ctx else "repl"), mode="exec")
    exec(code, ctx)
    return ctx[tree.body[0].name]


@pipes
def _is_platform_compatible(info: PyPI_Release, platform_tags: FrozenSet[PlatformTag], include_sdist=False) -> bool:

    if "py2" in info.justuse.python_tag:
        return False

    if not include_sdist and (".tar" in info.justuse.ext or info.justuse.python_tag in ("cpsource", "sdist")):
        return False

    our_python_tag = tags.interpreter_name() + tags.interpreter_version()
    python_tag = info.justuse.python_tag or "cp" + info.python_version.replace(".", "")
    if python_tag in ("py3", "cpsource"):
        python_tag = our_python_tag
    our_platform_tags = info.justuse.platform_tag.split(".") << map(PlatformTag) >> frozenset
    is_sdist = info.packagetype == "sdist" or info.python_version == "source" or info.justuse.abi_tag == "none"
    return our_python_tag == python_tag and (
        (is_sdist and include_sdist) or any(our_platform_tags.intersection(platform_tags))
    )
