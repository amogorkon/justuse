import use

use("olefile", version="0.46", modes=use.auto_install)

use("sys", modes=use.fatal_exceptions)

use(
    "numpy",
    version="1.19.3",
    hashes={"83af653bb92d1e248ccf5fdb05ccc934c14b936bcfe9b917dc180d3f00250ac6"},
    modes=use.auto_install | use.no_public_installation,
)
