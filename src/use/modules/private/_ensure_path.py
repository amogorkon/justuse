from pathlib import Path
from functools import reduce
from typing import Union
import furl

from ..Decorators import pipes


@pipes
def _ensure_path(value: Union[bytes, str, furl.Path, Path]) -> Path:
    if isinstance(value, (str, bytes)):
        return Path(value).absolute()
    if isinstance(value, furl.Path):
        return (
            Path.cwd(),
            value.segments << map(Path) << tuple << reduce(Path.__truediv__),
        ) << reduce(Path.__truediv__)
    return value
