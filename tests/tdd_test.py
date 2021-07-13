import os
import pytest
import requests
import sys
import use
import warnings
from unit_test import reuse
from pathlib import Path
from yarl import URL

if Path("use").is_dir(): os.chdir("..")
import_base = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(import_base))
warnings.filterwarnings("ignore", category=DeprecationWarning)

@pytest.mark.xfail(True, reason="in development")
def test_failing_test():
  assert False, "This test is expected to fail"

def test_succeeding_test():
  assert True, "This test is expected to succeed"

# Add in-progress tests here

