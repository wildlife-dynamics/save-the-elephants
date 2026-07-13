"""Tests for ecoscope_workflows_ext_ste.tasks.skip._skip.

Both `skip_toggle` and `skip_file` are registered via `wt_registry.register()`,
which is a no-op at call time, so they behave as plain Python functions here.

`SkipSentinel` (see `wt_task.skip`) is a marker class used as a return value
to signal "this task's execution was skipped" -- a global singleton instance
`SKIP_SENTINEL` is the canonical value tasks return. These tests focus on the
branching logic in `skip_file` that decides between returning a real value
and returning that sentinel, plus the trivial pass-through behavior of
`skip_toggle`.
"""

import pytest
from wt_task.skip import SKIP_SENTINEL, SkipSentinel

from ecoscope_workflows_ext_ste.tasks.skip._skip import skip_file, skip_toggle


class TestSkipToggle:
    """skip_toggle is a pure pass-through so multiple `skip_file` nodes can
    gate on one shared value."""

    def test_default_returns_true(self):
        assert skip_toggle() is True

    def test_explicit_true(self):
        assert skip_toggle(True) is True
        assert skip_toggle(enabled=True) is True

    def test_explicit_false(self):
        assert skip_toggle(False) is False
        assert skip_toggle(enabled=False) is False

    def test_return_type_is_bool(self):
        assert isinstance(skip_toggle(True), bool)
        assert isinstance(skip_toggle(False), bool)


class TestSkipFile:
    """skip_file returns the filename when enabled, or SKIP_SENTINEL when
    disabled -- exercising the actual branching logic around SkipSentinel."""

    def test_default_args_is_disabled_and_returns_sentinel(self):
        # enabled defaults to False, so by default the step is skipped.
        result = skip_file()
        assert result is SKIP_SENTINEL
        assert isinstance(result, SkipSentinel)

    def test_enabled_false_returns_sentinel_regardless_of_filename(self):
        result = skip_file(filename="custom.html", enabled=False)
        assert result is SKIP_SENTINEL

    def test_enabled_true_returns_default_filename(self):
        result = skip_file(enabled=True)
        assert result == "animated.html"
        assert not isinstance(result, SkipSentinel)

    def test_enabled_true_returns_custom_filename(self):
        result = skip_file(filename="my_video.html", enabled=True)
        assert result == "my_video.html"

    @pytest.mark.parametrize("filename", ["a.html", "", "nested/dir/file.html"])
    def test_enabled_true_various_filenames_pass_through(self, filename):
        assert skip_file(filename=filename, enabled=True) == filename

    def test_enabled_true_with_none_filename_returns_none_not_sentinel(self):
        # Optional[str] permits None; when enabled, the function returns
        # whatever filename it was given (including None) rather than the
        # sentinel -- so `None` and "skipped" are distinguishable here only
        # by identity/type check against SkipSentinel, not by truthiness.
        result = skip_file(filename=None, enabled=True)
        assert result is None
        assert not isinstance(result, SkipSentinel)

    def test_sentinel_identity_is_the_shared_singleton(self):
        first = skip_file(enabled=False)
        second = skip_file(filename="other.html", enabled=False)
        assert first is second is SKIP_SENTINEL

    def test_positional_args(self):
        assert skip_file("foo.html", True) == "foo.html"
        assert skip_file("foo.html", False) is SKIP_SENTINEL


class TestSkipSentinelType:
    """Sanity checks on the SkipSentinel type itself, since skip_file's
    contract depends on it."""

    def test_repr(self):
        assert repr(SKIP_SENTINEL) == "<SkipSentinel>"

    def test_is_instance_of_skip_sentinel(self):
        assert isinstance(SKIP_SENTINEL, SkipSentinel)

    def test_not_falsy_by_convention_but_is_a_distinct_object(self):
        # SkipSentinel doesn't define __bool__/__eq__, so identity/isinstance
        # checks are the correct way consumers should detect a skip, not
        # equality or truthiness comparisons.
        assert SKIP_SENTINEL != None  # noqa: E711
        assert SKIP_SENTINEL != False  # noqa: E712
        assert bool(SKIP_SENTINEL) is True
