import traceback

try:
    assert False, "foobar"
except AssertionError as exc:
    print(exc.__class__.__name__, exc)

#print(traceback.format_exception_only(exc, AssertionError))