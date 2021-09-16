from use import use

try:
    from typeguard.importhook import install_import_hook
    install_import_hook(packages=["use"])
except ImportError:
    # typeguard not installed
    pass
