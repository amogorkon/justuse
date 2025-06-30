"""Core tests to quickly check the basic functionality of the library on every edit."""

import json

import pytest

from justuse import Repo, use
from justuse.exceptions import JustUseError, RepoPathNotFoundError


def test_use_import_builtin():
    math = use("math")
    assert hasattr(math, "sqrt")
    assert math.sqrt(16) == 4


def test_justuseerror_json():
    try:
        raise JustUseError(
            message="Test error message",
            context={"foo": "bar"},
            recovery_actions=[{"type": "noop", "description": "No action"}],
            error_id="JU1234",
            severity="fatal",
            error_namespace="JUSTUSE_TEST",
        )
    except JustUseError as err:
        js = err.to_json()
        data = json.loads(js)
        assert data["error_id"] == "JU1234"
        assert data["type"] == "JustUseError"
        assert data["severity"] == "fatal"
        assert data["context"] == {"foo": "bar"}
        assert data["recovery_actions"][0]["type"] == "noop"
        assert data["error_namespace"] == "JUSTUSE_TEST"
        assert data["justuse_version"] == "2025.26"
        assert "timestamp" in data
        assert str(err).startswith("[JU1234]: Test error message")


def test_use_test_module_err():
    # Use the test_module.py from the amogorkon/justuse repo's tests directory
    # raises justuse.exceptions.RepoPathNotFoundError
    with pytest.raises(
        RepoPathNotFoundError,
    ):
        use(
            Repo.github(
                repo_name="amogorkon/justuse",
                path="tests/.tests/test_module.py",
                ref="main",
            )
        )


def test_use_test_module_succ():
    # Use the test_module.py from the amogorkon/justuse repo's tests directory
    mod = use(
        Repo.github(
            repo_name="amogorkon/justuse",
            path="tests/.tests/test_module.py",
            ref="unstable",
        )
    )
    assert mod.test_function() == "Test function executed successfully!"
