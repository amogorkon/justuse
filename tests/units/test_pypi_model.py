from use.pydantics import JustUse_Info, PyPI_Release


def test_pypi_model():
    release = PyPI_Release(
        comment_text="test",
        digests={"md5": "asdf"},
        url="https://files.pythonhost",
        ext=".whl",
        packagetype="bdist_wheel",
        distribution="numpy",
        requires_python=False,
        python_version="cp3",
        python_tag="cp3",
        platform_tag="cp4",
        filename="numpy-1.19.5-cp3-cp3-cp4-bdist_wheel.whl",
        abi_tag="cp3",
        yanked=False,
        version="1.19.5",
    )
    assert type(release)(**release.dict()) == release

    info = JustUse_Info(
        distribution="numpy",
        version="1.19.5",
        build_tag="cp4",
        python_tag="cp4",
        abi_tag="cp4",
        platform_tag="cp4",
        ext="whl",
    )
    assert type(info)(**info.dict()) == info
