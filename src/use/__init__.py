from typeguard.importhook import install_import_hook
install_import_hook(packages=["use", "tests"])

from use import use
