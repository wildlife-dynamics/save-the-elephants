"""Tests for ecoscope_workflows_ext_ste.tasks.filter._time.

Registers 5 functions via `wt_registry.register()` (a no-op at call time, so
they behave as plain Python functions here):

  - create_date_offset: builds a `DateOffsetSchema` from individual
    year/month/.../second components.
  - shift_period: shifts *both* `since` and `until` of a `TimeRange` back by
    an offset (the window length is preserved).
  - previous_period: returns the period immediately preceding
    `timerange.since`, of length `offset` (i.e. `until` becomes the old
    `since`, and the new `since` is `offset` before that).
  - get_duration: measures the length of a `TimeRange` in a given unit.
    Fixed-length units (seconds/minutes/hours/days/weeks) are exact ratios of
    total_seconds(); calendar units (months/years) use whole calendar units
    plus a fractional remainder measured against the *actual* length of the
    unit following the whole part (so results vary with the calendar, e.g.
    February vs. January).
  - flexible_previous_period: computes a comparison period ending at
    `timerange.since`, with the start controlled by one of three modes
    (Custom relativedelta, named Preset, or an explicit Calendar date).

`TimeRange` (from `ecoscope.platform.tasks.filter._filter`) requires `since`
and `until` to be both timezone-naive or both timezone-aware; if both are
naive it stamps them with the tz derived from the `timezone` field.
"""

from datetime import datetime, timedelta as dt_timedelta, timezone as dt_timezone

import pytest
from dateutil.relativedelta import relativedelta
from pydantic import ValidationError

from ecoscope.platform.tasks.filter._filter import TimeRange, UTC_TIMEZONEINFO
from ecoscope_workflows_ext_ste.tasks.filter._time import (
    CalendarPreviousPeriod,
    CustomPreviousPeriod,
    DateOffsetSchema,
    PresetPreviousPeriod,
    create_date_offset,
    flexible_previous_period,
    get_duration,
    previous_period,
    shift_period,
)

UTC = UTC_TIMEZONEINFO


def make_time_range(since, until, tz=UTC):
    return TimeRange(since=since, until=until, timezone=tz)


class TestTimeRangeTimezoneHandling:
    """Sanity-check the underlying `TimeRange` model's tz behavior, since
    every function under test depends on it."""

    def test_naive_datetimes_get_stamped_with_timezone_from_utc_offset(self):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        assert tr.since.tzinfo is not None
        assert tr.until.tzinfo is not None
        assert tr.since.utcoffset().total_seconds() == 0

    def test_aware_datetimes_are_accepted_unchanged(self):
        since = datetime(2024, 1, 1, tzinfo=dt_timezone.utc)
        until = datetime(2024, 2, 1, tzinfo=dt_timezone.utc)
        tr = make_time_range(since, until)
        assert tr.since == since
        assert tr.until == until

    def test_mixed_naive_and_aware_raises_validation_error(self):
        with pytest.raises(ValidationError):
            make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1, tzinfo=dt_timezone.utc))


class TestCreateDateOffset:
    def test_defaults_produce_zeroed_offset_with_n_one(self):
        result = create_date_offset()
        assert result == DateOffsetSchema(n=1)

    def test_all_components_are_captured(self):
        result = create_date_offset(n=2, years=1, months=2, weeks=3, days=4, hours=5, minutes=6, seconds=7)
        assert result.n == 2
        assert result.years == 1
        assert result.months == 2
        assert result.weeks == 3
        assert result.days == 4
        assert result.hours == 5
        assert result.minutes == 6
        assert result.seconds == 7

    def test_returns_date_offset_schema_instance(self):
        assert isinstance(create_date_offset(months=1), DateOffsetSchema)


class TestDateOffsetSchemaConversion:
    def test_to_offset_round_trips_through_from_offset(self):
        schema = DateOffsetSchema(n=3, months=2, days=5)
        off = schema.to_offset()
        back = DateOffsetSchema.from_offset(off)
        assert back == schema

    def test_to_offset_omits_zero_valued_kwargs(self):
        schema = DateOffsetSchema(months=2)
        off = schema.to_offset()
        # DateOffset.kwds should only contain the non-zero fields we set
        assert off.kwds == {"months": 2}

    def test_all_zero_component_offset_is_not_actually_a_no_op(self):
        # SUSPECTED BUG: `to_offset()` filters out zero-valued kwargs
        # (`if k != "n" and v`), so an all-zero DateOffsetSchema produces
        # `DateOffset(n=1)` with an *empty* kwds dict. Pandas special-cases
        # a `DateOffset` with no unit kwargs at all as a generic 1-day
        # offset (this differs from e.g. `DateOffset(n=1, months=0)`, which
        # IS a true no-op since `months` is present in kwds). So a "zero
        # offset" as built by this schema actually shifts by one day, not
        # zero, contradicting `_offset_description`'s own "zero offset"
        # label for this case. This test documents the actual behavior;
        # it is not asserting that the behavior is desirable.
        schema = DateOffsetSchema()
        off = schema.to_offset()
        assert off.kwds == {}

        import pandas as pd

        ts = pd.Timestamp("2024-06-15")
        assert ts - off == ts - pd.Timedelta(days=1)

    def test_contrast_with_pandas_date_offset_that_keeps_an_explicit_zero_kwarg(self):
        # Direct contrast showing *why* the above is surprising: if pandas'
        # own `DateOffset` is given an explicit `months=0` kwarg (i.e. the
        # kwarg is present in `.kwds`, just zero-valued), it IS a true
        # no-op -- unlike the empty-kwds case produced by
        # `DateOffsetSchema.to_offset()` above.
        import pandas as pd
        from pandas.tseries.offsets import DateOffset

        ts = pd.Timestamp("2024-06-15")
        assert ts - DateOffset(n=1, months=0) == ts


class TestShiftPeriod:
    def test_shifts_both_since_and_until_preserving_window_length(self):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        offset = create_date_offset(months=1)
        result = shift_period(tr, offset)

        assert result.since == datetime(2023, 12, 1, tzinfo=dt_timezone.utc)
        assert result.until == datetime(2024, 1, 1, tzinfo=dt_timezone.utc)
        # window length preserved
        assert (result.until - result.since) == (tr.until - tr.since)

    def test_preserves_timezone_and_time_format_fields(self):
        tr = TimeRange(since=datetime(2024, 1, 1), until=datetime(2024, 2, 1), timezone=UTC, time_format="%Y-%m-%d")
        result = shift_period(tr, create_date_offset(days=1))
        assert result.timezone == tr.timezone
        assert result.time_format == "%Y-%m-%d"

    def test_all_zero_offset_actually_shifts_by_one_day(self):
        # SUSPECTED BUG (see TestDateOffsetSchemaConversion above): an
        # all-zero `create_date_offset()` is not a true no-op when threaded
        # through `shift_period`, because `DateOffsetSchema.to_offset()`
        # turns it into an empty-kwds `DateOffset`, which pandas treats as a
        # generic 1-day offset. So `shift_period` with the "default" offset
        # shifts the period back by one day rather than leaving it alone.
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        result = shift_period(tr, create_date_offset())
        assert result.since == tr.since - dt_timedelta(days=1)
        assert result.until == tr.until - dt_timedelta(days=1)


class TestPreviousPeriod:
    def test_until_becomes_old_since_and_since_steps_back_by_offset(self):
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        offset = create_date_offset(months=1)
        result = previous_period(tr, offset)

        assert result.until == tr.since
        assert result.since == datetime(2024, 2, 1, tzinfo=dt_timezone.utc)

    def test_previous_period_window_length_matches_offset_not_original_window(self):
        # original window is 2 months (Jan->Mar) but offset is only 10 days:
        # previous_period's length should reflect the offset, not the
        # original timerange's length.
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 3, 1))
        offset = create_date_offset(days=10)
        result = previous_period(tr, offset)
        assert result.until == tr.since
        assert (result.until - result.since).days == 10


class TestGetDuration:
    @pytest.mark.parametrize(
        "unit, expected",
        [
            ("seconds", 2678400.0),
            ("minutes", 44640.0),
            ("hours", 744.0),
            ("days", 31.0),
            ("weeks", 31.0 * 86400 / 604800),
        ],
    )
    def test_fixed_length_units_are_exact_ratios_of_total_seconds(self, unit, expected):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        assert get_duration(tr, unit) == pytest.approx(expected)

    def test_whole_month_duration_is_exactly_one(self):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        assert get_duration(tr, "months") == pytest.approx(1.0)

    def test_whole_year_duration_is_exactly_one(self):
        tr = make_time_range(datetime(2023, 1, 1), datetime(2024, 1, 1))
        assert get_duration(tr, "years") == pytest.approx(1.0)

    def test_partial_month_duration_uses_actual_length_of_the_partial_unit(self):
        # Jan 1 -> Jan 16 is a partial month measured against January's 31 days.
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 1, 16))
        assert get_duration(tr, "months") == pytest.approx(15 / 31)

    def test_default_time_unit_is_months(self):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        assert get_duration(tr) == get_duration(tr, "months")

    def test_reversed_range_yields_negative_duration(self):
        tr = make_time_range(datetime(2024, 2, 1), datetime(2024, 1, 1))
        assert get_duration(tr, "days") == pytest.approx(-31.0)

    def test_invalid_time_unit_raises_value_error(self):
        tr = make_time_range(datetime(2024, 1, 1), datetime(2024, 2, 1))
        with pytest.raises(ValueError, match="invalid time_unit"):
            get_duration(tr, "fortnights")

    def test_zero_length_range_is_zero_in_every_unit(self):
        same = datetime(2024, 1, 1)
        tr = make_time_range(same, same)
        for unit in ("seconds", "minutes", "hours", "days", "weeks", "months", "years"):
            assert get_duration(tr, unit) == pytest.approx(0.0)


class TestFlexiblePreviousPeriodCustom:
    def test_custom_offset_subtracts_relativedelta_from_since(self):
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        result = flexible_previous_period(tr, CustomPreviousPeriod(months=1))
        assert result.until == tr.since
        assert result.since == datetime(2024, 2, 1, tzinfo=dt_timezone.utc)

    def test_custom_offset_supports_all_component_fields(self):
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        offset = CustomPreviousPeriod(years=1, months=1, weeks=1, days=1, hours=1, minutes=1, seconds=1)
        result = flexible_previous_period(tr, offset)
        expected_since = tr.since - relativedelta(years=1, months=1, weeks=1, days=1, hours=1, minutes=1, seconds=1)
        assert result.since == expected_since
        assert result.until == tr.since


class TestFlexiblePreviousPeriodPreset:
    def test_same_as_current_period_preserves_calendar_span_via_relativedelta(self):
        # "Same as current period" computes `relativedelta(until, since)` of
        # the *current* period and subtracts that from `since`. Because
        # relativedelta expresses the span in calendar units (here: exactly
        # 1 month), not a fixed timedelta, the resulting previous period can
        # have a different number of days than the original (e.g. stepping
        # from a 31-day March back a "1 month" relativedelta lands on
        # Feb 1 -> Mar 1, a 29-day span in this leap year) even though the
        # calendar-unit span is identical.
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        result = flexible_previous_period(tr, PresetPreviousPeriod(preset="Same as current period"))
        assert result.until == tr.since
        assert result.since == datetime(2024, 2, 1, tzinfo=dt_timezone.utc)
        # the actual day-count differs from the original window here
        assert (result.until - result.since) != (tr.until - tr.since)

    def test_same_as_current_period_matches_original_window_length_for_fixed_unit_span(self):
        # When the current period's span is expressed in fixed-length units
        # (here: exactly 10 days, with no whole months), relativedelta
        # degrades to a plain day delta, so the previous period's length
        # does match the original.
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 3, 11))
        result = flexible_previous_period(tr, PresetPreviousPeriod(preset="Same as current period"))
        assert result.until == tr.since
        assert (result.until - result.since) == (tr.until - tr.since)

    @pytest.mark.parametrize(
        "preset, months_back",
        [
            ("1 month back", 1),
            ("3 months back", 3),
            ("6 months back", 6),
        ],
    )
    def test_named_month_presets_step_back_from_since(self, preset, months_back):
        tr = make_time_range(datetime(2024, 6, 15), datetime(2024, 7, 15))
        result = flexible_previous_period(tr, PresetPreviousPeriod(preset=preset))
        assert result.until == tr.since
        assert result.since == tr.since - relativedelta(months=months_back)

    def test_one_year_back_preset(self):
        tr = make_time_range(datetime(2024, 6, 15), datetime(2024, 7, 15))
        result = flexible_previous_period(tr, PresetPreviousPeriod(preset="1 year back"))
        assert result.since == tr.since - relativedelta(years=1)

    def test_default_preset_is_one_month_back(self):
        assert PresetPreviousPeriod().preset == "1 month back"


class TestFlexiblePreviousPeriodCalendar:
    def test_calendar_since_is_used_verbatim_and_until_is_original_since(self):
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        calendar_since = datetime(2023, 1, 1, tzinfo=dt_timezone.utc)
        result = flexible_previous_period(tr, CalendarPreviousPeriod(since=calendar_since))
        assert result.since == calendar_since
        assert result.until == tr.since

    def test_calendar_since_naive_with_aware_timerange_raises_on_construction(self):
        tr = make_time_range(datetime(2024, 3, 1), datetime(2024, 4, 1))
        # tr.since/until are tz-aware (naive inputs get stamped by TimeRange);
        # pairing that with a naive calendar `since` violates TimeRange's
        # "both naive or both aware" invariant when building the result.
        with pytest.raises(ValidationError):
            flexible_previous_period(tr, CalendarPreviousPeriod(since=datetime(2023, 1, 1)))


class TestFlexiblePreviousPeriodPreservesMetadata:
    def test_timezone_and_time_format_are_carried_over(self):
        tr = TimeRange(since=datetime(2024, 3, 1), until=datetime(2024, 4, 1), timezone=UTC, time_format="%Y-%m-%d")
        result = flexible_previous_period(tr, CustomPreviousPeriod(months=1))
        assert result.timezone == tr.timezone
        assert result.time_format == "%Y-%m-%d"
