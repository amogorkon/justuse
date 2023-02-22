from typing import Any, NamedTuple

def typed(row, idx: int, kind: type, default: Any = ..., debugging=False):
    res = row[idx]
    if default is not ...:
        if res == "" or res is None:
            return default
        assert (
            type(res) is kind or res is None
        ), f"'{res}'"
        return res
    else:
        assert (
            type(res) is kind
        ), f"'{res}' ({type(res)}) is not {kind}! "
    return res

typed([""], 0, int, 2)