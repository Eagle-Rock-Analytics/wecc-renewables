"""
Test constants module.

This module contains unit tests for the constants module, which defines
sentinel values and constants used throughout the renewable-profiles package.
"""

from src.constants import UNSET


class TestConstants:
    """Test constants and sentinel values."""

    def test_unset_sentinel(self):
        """Test that UNSET is a unique sentinel object."""
        # UNSET should be a unique object
        assert UNSET is not None
        assert UNSET is not False
        assert UNSET is not True
        assert UNSET != 0
        assert UNSET != ""

        # Should be the same object when imported multiple times
        from src.constants import UNSET as UNSET2

        assert UNSET is UNSET2

        # Should have a reasonable string representation
        assert str(UNSET).startswith("<object object")

        # Should be usable in boolean context
        assert bool(UNSET) is True

    def test_unset_uniqueness(self):
        """Test that UNSET is unique compared to other sentinel objects."""
        other_sentinel = object()
        assert UNSET is not other_sentinel
        assert other_sentinel != UNSET

    def test_unset_as_default_parameter(self):
        """Test UNSET can be used as a default parameter."""

        def test_func(param=UNSET):
            return param is UNSET

        assert test_func() is True
        assert test_func(None) is False
        assert test_func("value") is False
