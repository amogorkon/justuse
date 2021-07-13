import os
import sys
import warnings
from pathlib import Path

import pytest
import requests
import use
from yarl import URL

from unit_test import reuse

if Path("use").is_dir(): os.chdir("..")
import_base = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(import_base))
warnings.filterwarnings("ignore", category=DeprecationWarning)

not_local = not Path("local_switch").exists()

@pytest.mark.xfail(not_local, reason="in development")
def test_failing_test():
  assert False, "This test is expected to fail"

def test_succeeding_test():
  assert True, "This test is expected to succeed"

# Add in-progress tests here

