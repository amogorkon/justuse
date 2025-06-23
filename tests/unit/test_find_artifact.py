from justuse.pimp import _get_project_from_pypi


def test_find_windows_artifact(reuse):
    assert (
        reuse.Version("3.17.3")
        in _get_project_from_pypi(package_name="protobuf").releases
    )
