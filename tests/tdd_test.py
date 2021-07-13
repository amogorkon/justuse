import os
import sys
import warnings
from pathlib import Path
from unittest import skip

import pytest
import requests
from yarl import URL

from unit_test import reuse

not_local = not Path("local_switch").exists()


# Add in-progress tests here

#@pytest.mark.xfail(not_local, reason="in development") 
# def test_registry_first_line_warning(reuse):
#   with open(reuse.home / "registry.json") as file:
#     assert file.readlines()[0].startswith("### WARNING")
