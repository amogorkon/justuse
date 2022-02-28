import io
from contextlib import redirect_stdout

with io.StringIO() as buf, redirect_stdout(buf):
    print("adsf")
    x = buf.getvalue()

