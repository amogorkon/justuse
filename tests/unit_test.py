import os, sys
here = os.path.split(os.path.abspath(os.path.dirname(__file__)))
src = os.path.join(here[0], "src")
sys.path.insert(0,src)

from unittest import TestCase, skip
import use
from pathlib import Path
from yarl import URL

def test_simple_path():
    foo_path = os.path.join(*here, "foo.py")
    print(f"loading foo module via use(Path({foo_path}))")
    mod = use(Path(foo_path), initial_globals={"a": 42})
    assert mod.test() == 42
    
def test_simple_url():
    foo_path = "/".join([*here, "foo.py"])
    cwd = os.getcwd()
    import http.server
    port = 8089
    svr = http.server.HTTPServer(
      ("", port), http.server.SimpleHTTPRequestHandler
    )
    foo_uri = f"http://localhost:{port}/tests/foo.py"
    print(f"starting thread to handle HTTP request on port {port}")
    import threading
    thd = threading.Thread(target=svr.handle_request)
    thd.start()
    print(f"loading foo module via use(URL({foo_uri}))")
    mod = use(URL(foo_uri), initial_globals={"a": 42})
    assert mod.test() == 42
