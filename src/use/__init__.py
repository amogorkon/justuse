try:
    # from typeguard.importhook import install_import_hook
    # TODO: Figure out how we might be able to re-enable at a later date
    pass
    # install_import_hook(packages=["use"])
except ImportError:
    # typeguard not installed
    pass

from use import use
