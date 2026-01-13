from enum import Enum
from datetime import datetime, timedelta
from pydantic import ConfigDict
from typing import Union, Annotated
from pydantic import BaseModel, Field
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.tasks.filter._filter import TimeRange, TimezoneInfo, UTC_TIMEZONEINFO, DEFAULT_TIME_FORMAT


class PreviousPeriodType(str, Enum):
    """Enum for previous period selection options."""

    SAME_AS_CURRENT = "Same as current period"
    PREVIOUS_MONTH = "Previous month"
    PREVIOUS_3_MONTHS = "Previous 3 months"
    PREVIOUS_6_MONTHS = "Previous 6 months"
    PREVIOUS_YEAR = "Previous year"


class PreviousTimeRange(BaseModel):
    """Custom TimeRange with UTC defaults for previous period selection."""

    since: Annotated[datetime, Field(description="The start time")]
    until: Annotated[datetime, Field(description="The end time")]
    timezone: Annotated[
        TimezoneInfo, Field(default=UTC_TIMEZONEINFO, description="Timezone information (defaults to UTC)")
    ] = UTC_TIMEZONEINFO
    time_format: Annotated[str, Field(default=DEFAULT_TIME_FORMAT, description="The time format")] = DEFAULT_TIME_FORMAT

    def to_time_range(self) -> TimeRange:
        """Convert to standard TimeRange object."""
        return TimeRange(since=self.since, until=self.until, timezone=self.timezone, time_format=self.time_format)


class PreviousCustomTimeRangeOption(BaseModel):
    model_config = ConfigDict(title="Choose which period you'd like to select your date from")

    custom: Annotated[PreviousPeriodType, Field(description="Select the previous period type")]


class PreviousTimeRangeOption(BaseModel):
    model_config = ConfigDict(title="Enter date range you'd like to compare the data to")
    time_range: PreviousTimeRange  # Now using PreviousTimeRange with defaults


# Union type for either custom or manual selection
PrevOption = Union[PreviousCustomTimeRangeOption, PreviousTimeRangeOption]


@task
def determine_previous_period(option: PrevOption, current_time_range: TimeRange) -> TimeRange:
    """
    Determines the previous period based on the selected option.

    Args:
        option: Either a custom period selection or manual TimeRange
        current_time_range: The current period being analyzed

    Returns:
        TimeRange representing the previous period for comparison
    """

    if isinstance(option, PreviousTimeRangeOption):
        # Convert PreviousTimeRange to standard TimeRange
        return option.time_range.to_time_range()

    if option.custom == PreviousPeriodType.SAME_AS_CURRENT:
        # Calculate the duration of the current period
        duration = current_time_range.until - current_time_range.since

        # Shift the entire period backwards by its own duration
        prev_since = current_time_range.since - duration
        prev_until = current_time_range.until - duration

    elif option.custom == PreviousPeriodType.PREVIOUS_MONTH:
        prev_until = current_time_range.since
        prev_since = prev_until - timedelta(days=30)

    elif option.custom == PreviousPeriodType.PREVIOUS_3_MONTHS:
        prev_until = current_time_range.since
        prev_since = prev_until - timedelta(days=90)

    elif option.custom == PreviousPeriodType.PREVIOUS_6_MONTHS:
        prev_until = current_time_range.since
        prev_since = prev_until - timedelta(days=180)

    elif option.custom == PreviousPeriodType.PREVIOUS_YEAR:
        prev_until = current_time_range.since
        prev_since = prev_until - timedelta(days=365)

    else:
        raise ValueError(f"Unknown custom option: {option.custom}")

    return TimeRange(
        since=prev_since,
        until=prev_until,
        timezone=current_time_range.timezone,
        time_format=current_time_range.time_format,
    )
