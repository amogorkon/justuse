"""
Module to hold the decorators and other utility functions used in justuse.
"""

import ast
from collections.abc import Callable
from enum import Enum, Flag, auto
import inspect
from itertools import takewhile
from textwrap import dedent
from typing import Any
import re


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
        node.right.args.insert(
            0 if isinstance(node.op, ast.RShift) else len(node.right.args), node.left
        )
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
        if isinstance(d, ast.Call)
        and d.func.id != "pipes"
        or isinstance(d, ast.Name)
        and d.id != "pipes"
    ]
    tree = _PipeTransformer().visit(tree)
    code = compile(
        tree, filename=(ctx["__file__"] if "__file__" in ctx else "repl"), mode="exec"
    )
    exec(code, ctx)
    return ctx[tree.body[0].name]


def _is_callable(thing):
    try:
        return callable(
            object.__getattribute__(type(thing), "__call__")
        )  # to even catch weird non-callable __call__ methods
    except AttributeError:
        return False


class ALL(Enum):
    methods = inspect.ismethod
    properties = inspect.isdatadescriptor
    functions = inspect.isfunction
    classes = inspect.isclass


class ModeFlags(Flag):
    RECURSIVE = auto()
    VERBOSE = auto()
    TRIAL = auto()
    INC_DUNDER = auto()
    DEFAULT = auto()


RECURSIVE, VERBOSE, TRIAL, INC_DUNDER, DEFAULT = ModeFlags


def apply(
    decorator: Callable,
    kind: ALL,
    /,
    check=_is_callable,
    pattern="",
    mode: ModeFlags = DEFAULT,
):
    def sugar(*things: list[Any]):
        visited = {id(obj) for obj in vars(object).values()}
        visited.add(id(type))
        for thing in things:
            if id(thing) in visited:
                continue
            for name in dir(thing):
                obj = getattr(thing, name, None)
                qualname = getattr(obj, "__qualname__", name)

                if (
                    obj is None
                    or id(obj) in visited
                    or not check(obj)
                    or not kind(obj)
                    or not re.match(pattern, name)
                    or (
                        INC_DUNDER in mode
                        and name.startswith("__")
                        and name.endswith("__")
                    )
                ):
                    continue

                try:
                    wrapped = decorator(obj)
                    setattr(thing, name, wrapped)
                except BaseException as exc:
                    if VERBOSE in mode:
                        print(
                            f"Failed to apply decorator {decorator} to {qualname}: {exc}"
                        )
                    wrapped = obj
                visited.add(id(wrapped))
                if VERBOSE in mode:
                    print(f"Applied decorator {decorator} to {qualname}")
        return things[0] if len(things) == 1 else things

    return sugar
