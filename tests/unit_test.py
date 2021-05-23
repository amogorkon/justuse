import os, sys
here = os.path.split(os.path.abspath(os.path.dirname(__file__)))
src = os.path.join(here[0], "src")
sys.path.insert(0,src)

from unittest import TestCase, skip
import use
from pathlib import Path
from yarl import URL

def test_simple_path():
    mod = use(Path("./testsubdir/foo.py"))
    assert mod.test() == 42
    
def test_simple_url():
    mod = use(URL())
    assert mod.test() == 42