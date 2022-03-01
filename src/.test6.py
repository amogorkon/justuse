import io
from contextlib import redirect_stdout

import use

name = "numpy"
version = "1.21.0"
hashes = {
    "e80fe25cba41c124d04c662f33f6364909b985f2eb5998aaa5ae4b9587242cce",
    "cf680682ad0a3bef56dae200dbcbac2d57294a73e5b0f9864955e7dd7c2c2491",
    "2ba579dde0563f47021dcd652253103d6fd66165b18011dce1a0609215b2791e",  # win64
    "cc367c86eb87e5b7c9592935620f22d13b090c609f1b27e49600cd033b529f54",
}

mod = use(
    name,
    version=version,
    hashes=hashes,
    modes=use.auto_install,
)
print(mod.__version__)
