from enum import Enum
from datetime import datetime, timedelta
from typing import Union, Annotated
from pydantic import BaseModel, ConfigDict, Field
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import (
    TimeRange,
    TimezoneInfo,
    UTC_TIMEZONEINFO,
    DEFAULT_TIME_FORMAT,
)


class PreviousPeriodType(str, Enum):
    """Enum for previous period selection options."""

    SAME_AS_CURRENT = "Same as current period"
    PREVIOUS_MONTH = "Previous month"
    PREVIOUS_3_MONTHS = "Previous 3 months"
    PREVIOUS_6_MONTHS = "Previous 6 months"
    PREVIOUS_YEAR = "Previous year"


class PreviousTimeRange(BaseModel):
    """
    Manual previous period selection.

    Only `since` is provided by the user. `until` is always derived
    from the current period's start when converting to a TimeRange,
    enforcing the invariant that the previous period ends exactly
    where the current one begins.
    """

    since: Annotated[datetime, Field(description="The start time of the previous period")]
    timezone: Annotated[
        TimezoneInfo,
        Field(default=UTC_TIMEZONEINFO, description="Timezone (defaults to UTC)"),
    ] = UTC_TIMEZONEINFO
    time_format: Annotated[str, Field(default=DEFAULT_TIME_FORMAT, description="The time format")] = DEFAULT_TIME_FORMAT

    def to_time_range(self, current_time_range: TimeRange) -> TimeRange:
        """
        Convert to a standard TimeRange, pinning `until` to the start
        of the current period.

        Args:
            current_time_range: The current period being analysed.

        Returns:
            TimeRange where until == current_time_range.since.
        """
        return TimeRange(
            since=self.since,
            until=current_time_range.since,
            timezone=self.timezone,
            time_format=self.time_format,
        )


class PreviousCustomTimeRangeOption(BaseModel):
    """User selects a predefined previous period via the enum."""

    model_config = ConfigDict(title="Choose which period you'd like to select your date from")
    custom: Annotated[PreviousPeriodType, Field(description="Select the previous period type")]


class PreviousTimeRangeOption(BaseModel):
    """User manually specifies a start date for the previous period."""

    model_config = ConfigDict(title="Enter the start date you'd like the previous period to begin from")
    time_range: PreviousTimeRange


# Union type for either enum-driven or manual selection
PrevOption = Union[PreviousCustomTimeRangeOption, PreviousTimeRangeOption]

PERIOD_OFFSETS: dict[PreviousPeriodType, timedelta] = {
    PreviousPeriodType.PREVIOUS_MONTH: timedelta(days=30),
    PreviousPeriodType.PREVIOUS_3_MONTHS: timedelta(days=90),
    PreviousPeriodType.PREVIOUS_6_MONTHS: timedelta(days=180),
    PreviousPeriodType.PREVIOUS_YEAR: timedelta(days=365),
}


@task
def determine_previous_period(option: PrevOption, current_time_range: TimeRange) -> TimeRange:
    """
    Determines the previous period based on the selected option.

    Invariant: the returned TimeRange's `until` is always equal to
    `current_time_range.since` — the previous period ends exactly
    where the current one begins, regardless of which path is taken.

    Args:
        option: Either a predefined period selection (enum-driven)
                or a manual start date.
        current_time_range: The current period being analysed.

    Returns:
        TimeRange representing the previous period for comparison.
    """
    # --- manual path: user picked a start date, until is derived ---
    if isinstance(option, PreviousTimeRangeOption):
        return option.time_range.to_time_range(current_time_range)

    # --- enum-driven path: resolve the offset, then apply uniformly ---
    period_type = option.custom

    if period_type == PreviousPeriodType.SAME_AS_CURRENT:
        # Dynamic: mirror the current window's own duration
        offset = current_time_range.until - current_time_range.since
    elif period_type in PERIOD_OFFSETS:
        # Fixed: look up from the dict
        offset = PERIOD_OFFSETS[period_type]
    else:
        raise ValueError(f"Unknown custom option: {period_type}")

    # until is always pinned to the start of the current period
    prev_until = current_time_range.since
    prev_since = prev_until - offset

    return TimeRange(
        since=prev_since,
        until=prev_until,
        timezone=current_time_range.timezone,
        time_format=current_time_range.time_format,
    )
