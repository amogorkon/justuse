"""
Configuration and home directory setup for justuse.
"""

import os
import tempfile
from pathlib import Path
from uuid import uuid4
from .pydantics import Configuration

sessionID = uuid4()
del uuid4

home = Path(os.getenv("JUSTUSE_HOME", str(Path.home() / ".justuse-python")))
try:
    home.mkdir(mode=0o755, parents=True, exist_ok=True)
except PermissionError:
    home = tempfile.mkdtemp(prefix="justuse_")

config = Configuration()
