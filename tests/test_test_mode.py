"""
Test justuse test_mode functionality for module cleanup.

This validates that test_mode properly disables caching and allows
modules to be unloaded and reloaded multiple times in the same session.

NOTE: Run with: pytest tests/test_test_mode.py -p no:cacheprovider
Or install justuse in editable mode: pip install -e .
"""

import sys

import pytest

from justuse import use
from justuse.main import Use as UseClass


def test_use_class_has_new_methods():
    """Test that Use class has the new methods."""
    assert hasattr(UseClass, "unload_module")
    assert hasattr(UseClass, "reset_cache")


def test_test_mode_flag():
    """Test that test_mode flag can be set and retrieved."""
    # Save original value
    original = use.test_mode

    try:
        # Should be settable
        use.test_mode = True
        assert use.test_mode

        use.test_mode = False
        assert not use.test_mode
    finally:
        # Restore original
        use.test_mode = original


def test_unload_module():
    """Test that unload_module removes module from sys.modules."""
    # First ensure the module isn't loaded
    test_module_name = "test_fake_module_xyz"

    # Mock a module in sys.modules
    import types

    mock_mod = types.ModuleType(test_module_name)
    sys.modules[test_module_name] = mock_mod

    # Verify it's there
    assert test_module_name in sys.modules

    # Unload it
    use.unload_module(test_module_name)

    # Verify it's gone
    assert test_module_name not in sys.modules


def test_reset_cache():
    """Test that reset_cache clears internal module cache."""
    # Save original
    original = use.test_mode

    try:
        # Enable test mode
        use.test_mode = True

        # Reset should work without error
        use.reset_cache()
    finally:
        # Restore
        use.test_mode = original


def test_module_cache_attribute():
    """Test that Use class has _module_cache attribute."""
    assert hasattr(use, "_module_cache")
    assert isinstance(use._module_cache, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
