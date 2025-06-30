import pytest

from justuse import Repo, assumption, use
from justuse.repo import GitHubRepo


def test_github_repo_creation():
    repo = Repo.github("amogorkon/justuse", "docs/demo.py")
    assert assumption(repo, GitHubRepo)
    assert repo.repo_name == "amogorkon/justuse"
    assert repo.path == "docs/demo.py"
    assert repo.ref == "main"  # Default ref
    assert repo.subdir is None


def test_github_repo_url():
    repo = Repo.github("amogorkon/justuse", "docs/demo.py")
    expected_url = "https://github.com/amogorkon/justuse/blob/main/docs/demo.py"
    assert repo.url() == expected_url

    repo_with_subdir = Repo.github(
        "amogorkon/justuse", "docs/demo.py", subdir="examples"
    )
    expected_url_with_subdir = (
        "https://github.com/amogorkon/justuse/blob/main/docs/demo.py/examples"
    )
    assert repo_with_subdir.url() == expected_url_with_subdir


@pytest.mark.xfail(reason="GitHubRepo functionality is under development")
def test_github_repo_import():
    # Import the module via `use` with Repo.github
    imported_module = use(
        Repo.github("amogorkon/justuse", "tests/.tests/test_module.py", ref="unstable")
    )

    # Execute the function and assert its return value
    result = imported_module.test_function()
    assert result == "Test function executed successfully!"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
