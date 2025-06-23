def test_nirvana(reuse):
    with reuse.raises(reuse.NirvanaWarning):
        reuse()
