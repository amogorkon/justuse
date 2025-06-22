from enum import Enum
from typing import NamedTuple
from warnings import warn
from use.aspectizing import apply_aspect

version = 1

class UserMessage:
    def not_reloadable(name):
        return f"Beware {name} also contains non-function objects, it may not be safe to reload!"

class StrMessage(UserMessage):
    pass

apply_aspect(UserMessage, staticmethod)


raise ImportError(StrMessage.not_reloadable("foo"))
