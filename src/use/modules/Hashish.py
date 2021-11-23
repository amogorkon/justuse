import hashlib
from enum import Enum


class Hash(Enum):
    sha256 = hashlib.sha256
