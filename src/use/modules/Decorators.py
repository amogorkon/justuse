"""
Module to hold the decorators used in justuse


Author: ...
Date: 21.Oct.2021
"""

import ast
import inspect
from itertools import takewhile
from textwrap import dedent

from functools import singledispatch, update_wrapper
from typing import Callable

from private._PipeTransformer import _PipeTransformer


# singledispatch for methods
def methdispatch(func) -> Callable:
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        # so we can dispatch on None
        if len(args) == 1:
            args = args + (None,)
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


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
