import io
from contextlib import redirect_stdout

import use as reuse

name = "package-example"
with io.StringIO() as buf, redirect_stdout(buf):
    try:
        mod = reuse(name, modes=reuse.auto_install)
        assert False, f"Actually returned mod: {mod}"
    except RuntimeWarning:
        version = buf.getvalue().splitlines()[-1].strip()
    try:
        mod = reuse(name, version=version, modes=reuse.auto_install)
        assert False, f"Actually returned mod: {mod}"
    except RuntimeWarning:
        recommended_hash = buf.getvalue().splitlines()[-1].strip()
    mod = reuse(name, version=version, hashes=eval(recommended_hash), modes=reuse.auto_install)
    assert mod


# mod = use("package-example", version="0.1", hashes={"Y復㝿浯䨩䩯鷛㬉鼵爔滥哫鷕逮愁墕萮緩"}, modes=use.auto_install)
print(mod.some_var)
