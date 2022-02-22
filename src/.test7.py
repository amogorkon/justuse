import use

mod = use(
    "python-pypi-example",
    version="0.0.5",
    hashes="0c05ab37cd62568e6902b83d0231173d5ef8be09d832065122513dac60aeb81d",
    modes=use.auto_install,
    default=None,
)
print(mod)

mod = use(
    "python-pypi-example",
    version="0.0.3",
    hashes="ebb0f02d9f3ea3790f52a157f16089d434d2042adda11541f65e8eb2555abd6a",
    modes=use.auto_install,
)
print(mod)
