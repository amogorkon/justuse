import use

mod = use(
    "pygame",
    version="2.1.0",
    hashes={"232e51104db0e573221660d172af8e6fc2c0fda183c5dbf2aa52170f29aa9ec9"},
    modes=use.auto_install,
)
print(mod.__version__)
