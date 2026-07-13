"""Tests for ecoscope_workflows_ext_ste.tasks.config._set_values.

`set_numerical_var` is registered via `wt_registry.register()`, which is a
no-op at call time (it only records schema metadata in a global registry at
import time and returns the original function unchanged). That means calling
`set_numerical_var` in tests is exactly like calling any ordinary Python
function -- the `Annotated[Union[int, float], Field(...)]` signature is *not*
enforced by pydantic at runtime here (there is no `validate_call` wrapper),
so no coercion happens. Several tests below verify that empirically.
"""

import math

import pytest

from ecoscope_workflows_ext_ste.tasks.config._set_values import set_numerical_var


class TestIdentityBehavior:
    """The function is a pure identity function: whatever comes in goes out."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, -1, 42, -42, 10**9, -(10**9)],
    )
    def test_int_identity(self, value):
        result = set_numerical_var(value)
        assert result == value
        assert isinstance(result, int) and not isinstance(result, bool)

    @pytest.mark.parametrize(
        "value",
        [0.0, 1.5, -1.5, 3.14159, -0.0, 1e300, -1e300],
    )
    def test_float_identity(self, value):
        result = set_numerical_var(value)
        assert result == value
        assert isinstance(result, float)

    def test_returns_same_object_identity(self):
        # since the implementation is `return var`, identity (is) holds for
        # values that aren't small-int cached objects too, e.g. a float.
        value = 123.456
        result = set_numerical_var(value)
        assert result is value

    def test_int_boundary_zero(self):
        assert set_numerical_var(0) == 0
        result = set_numerical_var(0)
        assert isinstance(result, int) and not isinstance(result, bool)

    def test_float_nan_roundtrips(self):
        result = set_numerical_var(float("nan"))
        assert math.isnan(result)
        assert isinstance(result, float)

    def test_float_infinity_roundtrips(self):
        assert set_numerical_var(float("inf")) == float("inf")
        assert set_numerical_var(float("-inf")) == float("-inf")


class TestGenericOverIntAndFloat:
    """Confirm the function genuinely handles both numeric types distinctly."""

    def test_int_stays_int_not_promoted_to_float(self):
        result = set_numerical_var(7)
        assert isinstance(result, int) and not isinstance(result, bool)

    def test_float_stays_float_not_demoted_to_int(self):
        result = set_numerical_var(7.0)
        assert isinstance(result, float)

    @pytest.mark.parametrize(
        "value, expected_type",
        [
            (5, int),
            (5.0, float),
        ],
    )
    def test_type_preserved_for_equal_values(self, value, expected_type):
        # 5 == 5.0 but the function must not silently normalize the type
        result = set_numerical_var(value)
        assert type(result) is expected_type


class TestNoRuntimeCoercion:
    """Because `register()` performs no validation at call time, values that
    pydantic *would* coerce under a Union[int, float] field (e.g. bool,
    numeric strings) are passed through untouched by this plain function
    call -- verified empirically rather than assumed.
    """

    def test_bool_is_not_coerced_to_int(self):
        # bool is technically a subclass of int in Python, but since no
        # pydantic validation runs here, True/False pass straight through
        # as bool, they are not normalized to 1/0 ints.
        result = set_numerical_var(True)
        assert result is True
        assert isinstance(result, bool)

        result_false = set_numerical_var(False)
        assert result_false is False
        assert isinstance(result_false, bool)

    def test_numeric_string_is_not_coerced_to_number(self):
        # A pydantic-validated Union[int, float] field would coerce "5" to
        # 5. This bare function does not: it returns the original string.
        result = set_numerical_var("5")
        assert result == "5"
        assert isinstance(result, str)

    def test_non_numeric_input_is_not_rejected(self):
        # No validation is performed, so even a value that violates the
        # declared type hint passes straight through instead of raising.
        sentinel = object()
        assert set_numerical_var(sentinel) is sentinel
