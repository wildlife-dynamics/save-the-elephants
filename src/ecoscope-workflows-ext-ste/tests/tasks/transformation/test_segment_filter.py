"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._segment_filter.

`trajectory_segment_filter` is registered via `wt_registry.register()`,
which is a no-op at call time, so it behaves as a plain Python function
here: no pydantic validation happens on the *task's own* parameters
(min/max floats), only the explicit `if min > max: raise ValueError(...)`
guards written in its body run against those.

However, the task's *return value* is a `TrajectorySegmentFilter` pydantic
model (from `ecoscope.platform.tasks.preprocessing._preprocessing`), and
*that* model applies its own independent validation once constructed:
    - field-level constraints (checked by reading the installed model):
        min_length_meters: ge=0.001
        max_length_meters: gt=0.001
        min_time_secs:      ge=1
        max_time_secs:      gt=1
        min_speed_kmhr:     gt=0.001
        max_speed_kmhr:     gt=0.001
    - a `model_validator(mode="after")` that requires each max to be
      *strictly greater* than its corresponding min:
        max_length_meters > min_length_meters
        max_time_secs > min_time_secs
        max_speed_kmhr > min_speed_kmhr

This produces a notable interaction worth pinning down: the task's own
guard only raises when `min > max` (strict greater-than on the min side),
but the model's validator requires strict inequality in the other
direction too (rejects `min == max`). So passing equal min/max values
sails past the task's manual check but is then rejected by pydantic's
`ValidationError` inside the model constructor -- two different exception
types can surface from what looks like one validation concern, depending
on exactly how the equal/violating values are supplied. This is not fixed
here (the task file was explicitly off-limits), just documented and
exercised below.
"""

import pytest
from pydantic import ValidationError

from ecoscope_workflows_ext_ste.tasks.transformation._segment_filter import (
    trajectory_segment_filter,
)
from ecoscope.platform.tasks.preprocessing._preprocessing import (
    TrajectorySegmentFilter,
)


class TestDefaults:
    def test_default_call_returns_trajectory_segment_filter(self):
        result = trajectory_segment_filter()
        assert isinstance(result, TrajectorySegmentFilter)

    def test_default_values_match_task_signature_defaults(self):
        result = trajectory_segment_filter()
        assert result.min_length_meters == 0.001
        assert result.max_length_meters == 5000
        assert result.min_time_secs == 1
        assert result.max_time_secs == 21600
        assert result.min_speed_kmhr == 0.01
        assert result.max_speed_kmhr == 9.0


class TestCustomOverrides:
    def test_override_length_bounds(self):
        result = trajectory_segment_filter(min_length_meters=10, max_length_meters=100)
        assert result.min_length_meters == 10
        assert result.max_length_meters == 100

    def test_override_time_bounds(self):
        result = trajectory_segment_filter(min_time_secs=5, max_time_secs=3600)
        assert result.min_time_secs == 5
        assert result.max_time_secs == 3600

    def test_override_speed_bounds(self):
        result = trajectory_segment_filter(min_speed_kmhr=0.5, max_speed_kmhr=50)
        assert result.min_speed_kmhr == 0.5
        assert result.max_speed_kmhr == 50

    def test_all_fields_overridden_together(self):
        result = trajectory_segment_filter(
            min_length_meters=1,
            max_length_meters=2000,
            min_time_secs=2,
            max_time_secs=7200,
            min_speed_kmhr=1,
            max_speed_kmhr=20,
        )
        assert result.min_length_meters == 1
        assert result.max_length_meters == 2000
        assert result.min_time_secs == 2
        assert result.max_time_secs == 7200
        assert result.min_speed_kmhr == 1
        assert result.max_speed_kmhr == 20


class TestTaskLevelGuardClauses:
    """The explicit `if min > max: raise ValueError(...)` checks written in
    the task body, which run *before* the TrajectorySegmentFilter model is
    even constructed."""

    def test_min_length_greater_than_max_length_raises_value_error(self):
        with pytest.raises(ValueError, match="min_length_meters"):
            trajectory_segment_filter(min_length_meters=100, max_length_meters=1)

    def test_min_time_greater_than_max_time_raises_value_error(self):
        with pytest.raises(ValueError, match="min_time_secs"):
            trajectory_segment_filter(min_time_secs=100, max_time_secs=1)

    def test_min_speed_greater_than_max_speed_raises_value_error(self):
        with pytest.raises(ValueError, match="min_speed_kmhr"):
            trajectory_segment_filter(min_speed_kmhr=100, max_speed_kmhr=1)

    def test_error_message_includes_both_values(self):
        with pytest.raises(ValueError) as excinfo:
            trajectory_segment_filter(min_length_meters=100, max_length_meters=1)
        message = str(excinfo.value)
        assert "100" in message
        assert "1" in message

    def test_task_guard_is_not_a_pydantic_validation_error(self):
        # These are plain ValueErrors raised directly in the task body, not
        # pydantic ValidationErrors -- they fire before TrajectorySegmentFilter
        # is ever constructed.
        with pytest.raises(ValueError) as excinfo:
            trajectory_segment_filter(min_time_secs=100, max_time_secs=1)
        assert not isinstance(excinfo.value, ValidationError)


class TestEqualMinMaxBoundary:
    """min == max passes the task's own `min > max` guard (False) but is
    then rejected by the model's stricter `max > min` validator."""

    def test_equal_length_bounds_raise_pydantic_validation_error(self):
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_length_meters=100, max_length_meters=100)

    def test_equal_time_bounds_raise_pydantic_validation_error(self):
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_time_secs=100, max_time_secs=100)

    def test_equal_speed_bounds_raise_pydantic_validation_error(self):
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_speed_kmhr=5, max_speed_kmhr=5)


class TestUnderlyingModelFieldConstraints:
    """Values that pass the task's manual `min > max` guard but violate the
    TrajectorySegmentFilter model's own field constraints (ge/gt), and so
    surface as a pydantic ValidationError raised out of the model
    constructor inside the task."""

    def test_min_length_below_ge_floor_raises_validation_error(self):
        # 0.0005 < 0.001 violates min_length_meters' ge=0.001, but
        # 0.0005 < 0.0008 satisfies the task's own min>max guard.
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_length_meters=0.0005, max_length_meters=0.0008)

    def test_min_time_below_ge_floor_raises_validation_error(self):
        # 0.5 < 1 violates min_time_secs' ge=1, but 0.5 < 10 satisfies the
        # task's own guard.
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_time_secs=0.5, max_time_secs=10)

    def test_min_speed_at_or_below_gt_floor_raises_validation_error(self):
        # 0.0005 is below the exclusive minimum (gt=0.001) for
        # min_speed_kmhr, while still satisfying the task's own guard
        # against max_speed_kmhr=1.
        with pytest.raises(ValidationError):
            trajectory_segment_filter(min_speed_kmhr=0.0005, max_speed_kmhr=1)


class TestReturnTypeIsReusable:
    def test_returned_object_is_a_pydantic_basemodel_instance(self):
        result = trajectory_segment_filter()
        assert hasattr(result, "model_dump")

    def test_two_calls_with_same_args_produce_equal_but_distinct_objects(self):
        first = trajectory_segment_filter(min_length_meters=1, max_length_meters=10)
        second = trajectory_segment_filter(min_length_meters=1, max_length_meters=10)
        assert first == second
        assert first is not second
