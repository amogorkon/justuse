"""Core tests to quickly check the basic functionality of the library on every edit."""

from justuse import use
import json
from justuse.exceptions import JustUseError


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
        assert data["justuse_version"] == "0.9.1.1.0"
        assert "timestamp" in data
        assert str(err).startswith("[JU1234] JustUseError: Test error message")
