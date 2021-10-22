from ...pypi_model import PyPI_Project


def _filter_by_version(project: PyPI_Project, version: str) -> PyPI_Project:
    return PyPI_Project(
        **{
            **project.dict(),
            **{
                "releases": {version: [v.dict() for v in project.releases[version]]}
                if project.releases.get(version)
                else {}
            },
        }
    )
