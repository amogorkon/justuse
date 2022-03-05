import use

mod = use(
    "numpy",
    version="1.21.0",
    modes=use.auto_install,
)
print(mod.__version__)
