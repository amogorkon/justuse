mod = None
import traceback

try:
    mod = 3
    raise AssertionError
except Exception as e:
    print("foo")
    bar = traceback.format_exc()
