from justuse import auto_install, no_cleanup, use


def test_451_ignore_spaces_in_hashes():
    # single hash
    mod = use(
        "package-example",
        version="0.1",
        hashes="Y復㝿浯䨩䩯鷛㬉鼵爔滥哫鷕逮 愁墕萮緩",
        modes=auto_install | no_cleanup,
    )
    assert mod
    del mod
    # hash list
    mod = use(
        "package-example",
        version="0.1",
        hashes={"Y復㝿浯䨩䩯鷛㬉鼵爔滥哫鷕逮 愁墕萮緩"},
        modes=auto_install | no_cleanup,
    )
    assert mod
    del mod
