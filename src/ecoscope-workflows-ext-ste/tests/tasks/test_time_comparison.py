import pytest
from datetime import datetime, timedelta

from ecoscope_workflows_core.tasks.filter._filter import TimeRange, TimezoneInfo
from ecoscope_workflows_ext_ste.tasks._time_comparison import (
    determine_previous_period,
    PreviousCustomTimeRangeOption,
    PreviousTimeRangeOption,
    PreviousTimeRange,
    PreviousPeriodType,
)
from pydantic import ValidationError

UTC_TIMEZONEINFO = TimezoneInfo(label="UTC", tzCode="UTC", name="UTC", utc_offset="+00:00")


def make_timerange(since: datetime, until: datetime):
    return TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO)


def test_manual_time_range_is_returned_as_is():
    # Create PreviousTimeRange (has timezone/time_format defaults)
    manual = PreviousTimeRange(
        since=datetime(2024, 1, 1),
        until=datetime(2024, 1, 31),
        # timezone/time_format auto-default to UTC/DEFAULT_TIME_FORMAT
    )

    option = PreviousTimeRangeOption(time_range=manual)
    current = make_timerange(datetime(2025, 1, 1), datetime(2025, 1, 31))

    result = determine_previous_period(option, current)

    # Compare after conversion (since function returns TimeRange)
    assert result == manual.to_time_range()


def test_same_as_current_shifts_entire_period_backwards():
    current = make_timerange(
        datetime(2025, 2, 1),
        datetime(2025, 2, 11),  # 10 days
    )

    option = PreviousCustomTimeRangeOption(custom=PreviousPeriodType.SAME_AS_CURRENT)

    result = determine_previous_period(option, current)

    duration = current.until - current.since

    assert result.until == current.until - duration
    assert result.since == current.since - duration
    assert result.timezone == current.timezone


def test_previous_month_uses_30_day_window():
    current = make_timerange(
        datetime(2025, 3, 1),
        datetime(2025, 3, 31),
    )

    option = PreviousCustomTimeRangeOption(custom=PreviousPeriodType.PREVIOUS_MONTH)

    result = determine_previous_period(option, current)

    assert result.until == current.since
    assert result.since == current.since - timedelta(days=30)


def test_previous_3_months_uses_90_days():
    current = make_timerange(
        datetime(2025, 6, 1),
        datetime(2025, 6, 30),
    )

    option = PreviousCustomTimeRangeOption(custom=PreviousPeriodType.PREVIOUS_3_MONTHS)

    result = determine_previous_period(option, current)

    assert result.until == current.since
    assert result.since == current.since - timedelta(days=90)


def test_previous_6_months_uses_180_days():
    current = make_timerange(
        datetime(2025, 7, 1),
        datetime(2025, 7, 31),
    )

    option = PreviousCustomTimeRangeOption(custom=PreviousPeriodType.PREVIOUS_6_MONTHS)

    result = determine_previous_period(option, current)

    assert result.until == current.since
    assert result.since == current.since - timedelta(days=180)


def test_previous_year_uses_365_days():
    current = make_timerange(
        datetime(2025, 1, 1),
        datetime(2025, 12, 31),
    )

    option = PreviousCustomTimeRangeOption(custom=PreviousPeriodType.PREVIOUS_YEAR)

    result = determine_previous_period(option, current)

    assert result.until == current.since
    assert result.since == current.since - timedelta(days=365)


def test_unknown_custom_option_raises_error():
    make_timerange(
        datetime(2025, 1, 1),
        datetime(2025, 1, 31),
    )

    # This line already raises ValidationError due to invalid enum value
    with pytest.raises(ValidationError) as exc_info:
        PreviousCustomTimeRangeOption(
            custom="INVALID_OPTION"  # type: ignore
        )

    # Optional: assert the error is about the 'custom' field and enum
    assert "custom" in str(exc_info.value)
    assert "enum" in str(exc_info.value) or "should be" in str(exc_info.value)
