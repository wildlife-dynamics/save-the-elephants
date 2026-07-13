"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._image_matching.

Both `match_images_to_events` and `get_unmatched_images` are registered via
`wt_registry.register()`, which is a no-op at call time, so they behave as
plain Python functions here.

`match_images_to_events` pairs aerial-survey images to EarthRanger events by
timestamp proximity: for each event, every image within
[event_time - window, upper] is collected, where `upper` is
`created_at + window` when `created_at` is later than `event_time` (the "ER
Mobile" case: event started, photos taken in the field, then submitted later)
and `event_time + window` otherwise. Only events with at least one matched
image appear in the result.

`get_unmatched_images` diffs the full (timestamped) image set against the
paths already claimed in a `match_images_to_events` result.
"""

import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.transformation._image_matching import (
    _match_window,
    get_unmatched_images,
    match_images_to_events,
)

EVENT_TIME = pd.Timestamp("2024-06-01 10:00:00", tz="UTC")


def _images(rows):
    """Build an images_df from (file_path, datetime) pairs."""
    return pd.DataFrame(rows, columns=["file_path", "datetime"])


def _events(rows, columns=("id", "serial_number", "time", "created_at", "event_type", "event_type_display")):
    return pd.DataFrame(rows, columns=list(columns))


@pytest.fixture
def boundary_images_df():
    """Images placed exactly at/around a +/-4 minute window boundary."""
    return _images(
        [
            ("early.jpg", EVENT_TIME - pd.Timedelta(minutes=5)),
            ("lower.jpg", EVENT_TIME - pd.Timedelta(minutes=4)),
            ("ontime.jpg", EVENT_TIME),
            ("upper.jpg", EVENT_TIME + pd.Timedelta(minutes=4)),
            ("late.jpg", EVENT_TIME + pd.Timedelta(minutes=5)),
            ("no_dt.jpg", pd.NaT),
        ]
    )


@pytest.fixture
def single_event_df():
    return _events(
        [(1, "SN1", EVENT_TIME, pd.NaT, "carcass", "Carcass")],
    )


class TestMatchWindow:
    """Direct tests of the private `_match_window` helper: it defines the
    inclusive [event_time - window, upper] boundary semantics that
    `match_images_to_events` relies on."""

    def test_inclusive_lower_bound(self):
        images_df = _images([("a.jpg", EVENT_TIME - pd.Timedelta(minutes=4))])
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4))
        assert result == ["a.jpg"]

    def test_inclusive_upper_bound_no_created_at(self):
        images_df = _images([("a.jpg", EVENT_TIME + pd.Timedelta(minutes=4))])
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4))
        assert result == ["a.jpg"]

    def test_just_outside_lower_bound_excluded(self):
        images_df = _images([("a.jpg", EVENT_TIME - pd.Timedelta(minutes=4, seconds=1))])
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4))
        assert result == []

    def test_just_outside_upper_bound_excluded(self):
        images_df = _images([("a.jpg", EVENT_TIME + pd.Timedelta(minutes=4, seconds=1))])
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4))
        assert result == []

    def test_created_at_after_event_time_extends_upper_bound(self):
        created_at = EVENT_TIME + pd.Timedelta(minutes=10)
        images_df = _images([("a.jpg", EVENT_TIME + pd.Timedelta(minutes=13))])
        # Without created_at this would be excluded (13 > 4); with created_at
        # the upper bound becomes created_at + window = +14 min.
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4), created_at=created_at)
        assert result == ["a.jpg"]

    def test_created_at_before_event_time_does_not_extend_upper_bound(self):
        created_at = EVENT_TIME - pd.Timedelta(minutes=10)
        images_df = _images([("a.jpg", EVENT_TIME + pd.Timedelta(minutes=13))])
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4), created_at=created_at)
        assert result == []

    def test_preserves_images_df_order(self):
        images_df = _images(
            [
                ("second.jpg", EVENT_TIME + pd.Timedelta(minutes=1)),
                ("first.jpg", EVENT_TIME - pd.Timedelta(minutes=1)),
            ]
        )
        result = _match_window(EVENT_TIME, images_df, pd.Timedelta(minutes=4))
        assert result == ["second.jpg", "first.jpg"]


class TestMatchImagesToEvents:
    def test_matches_images_within_default_window_inclusive_bounds(self, boundary_images_df, single_event_df):
        result = match_images_to_events(boundary_images_df, single_event_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["matched_images"] == ["lower.jpg", "ontime.jpg", "upper.jpg"]
        assert row["image_count"] == 3

    def test_custom_time_window_minutes(self, boundary_images_df, single_event_df):
        result = match_images_to_events(boundary_images_df, single_event_df, time_window_minutes=5.0)

        row = result.iloc[0]
        assert set(row["matched_images"]) == {"early.jpg", "lower.jpg", "ontime.jpg", "upper.jpg", "late.jpg"}
        assert row["image_count"] == 5

    def test_images_without_datetime_are_dropped_before_matching(self, boundary_images_df, single_event_df):
        result = match_images_to_events(boundary_images_df, single_event_df)
        matched = result.iloc[0]["matched_images"]
        assert "no_dt.jpg" not in matched

    def test_event_with_no_matched_images_excluded_from_result(self):
        images_df = _images([("far.jpg", EVENT_TIME + pd.Timedelta(hours=5))])
        events_df = _events([(1, "SN1", EVENT_TIME, pd.NaT, "carcass", "Carcass")])

        result = match_images_to_events(images_df, events_df)
        assert result.empty

    def test_multiple_events_only_matched_ones_returned(self, boundary_images_df):
        events_df = _events(
            [
                (1, "SN1", EVENT_TIME, pd.NaT, "carcass", "Carcass"),
                (2, "SN2", EVENT_TIME + pd.Timedelta(hours=6), pd.NaT, "fire", "Fire"),
            ]
        )
        result = match_images_to_events(boundary_images_df, events_df)

        assert len(result) == 1
        assert result.iloc[0]["event_id"] == 1

    def test_created_at_after_event_time_extends_match_window(self):
        images_df = _images(
            [
                ("a.jpg", EVENT_TIME + pd.Timedelta(minutes=6)),
                ("b.jpg", EVENT_TIME + pd.Timedelta(minutes=20)),
            ]
        )
        events_df = _events([(10, "SN1", EVENT_TIME, EVENT_TIME + pd.Timedelta(minutes=10), "carcass", "Carcass")])
        result = match_images_to_events(images_df, events_df)

        assert len(result) == 1
        assert result.iloc[0]["matched_images"] == ["a.jpg"]

    def test_created_at_before_event_time_does_not_extend_window(self):
        images_df = _images([("a.jpg", EVENT_TIME + pd.Timedelta(minutes=6))])
        events_df = _events([(11, "SN1", EVENT_TIME, EVENT_TIME - pd.Timedelta(minutes=10), "carcass", "Carcass")])
        result = match_images_to_events(images_df, events_df)
        assert result.empty

    def test_missing_created_at_column_treated_as_none(self):
        images_df = _images([("a.jpg", EVENT_TIME)])
        events_df = pd.DataFrame([{"id": 1, "time": EVENT_TIME}])  # no "created_at" column at all
        result = match_images_to_events(images_df, events_df)

        assert len(result) == 1
        assert pd.isna(result.iloc[0]["created_at"])

    def test_missing_id_column_falls_back_to_row_index(self):
        images_df = _images([("a.jpg", EVENT_TIME)])
        events_df = pd.DataFrame([{"time": EVENT_TIME}])
        result = match_images_to_events(images_df, events_df)

        assert result.iloc[0]["event_id"] == 0

    def test_missing_optional_display_columns_become_none(self):
        images_df = _images([("a.jpg", EVENT_TIME)])
        events_df = pd.DataFrame([{"id": 1, "time": EVENT_TIME}])
        result = match_images_to_events(images_df, events_df)

        row = result.iloc[0]
        assert row["serial_number"] is None
        assert row["event_type"] is None
        assert row["event_type_display"] is None

    def test_tz_naive_event_time_is_localized_to_utc(self):
        # event["time"] has no tzinfo; the function should localize it to UTC
        # before comparing against tz-aware image datetimes.
        naive_time = pd.Timestamp("2024-06-01 10:00:00")  # no tz
        images_df = _images([("a.jpg", EVENT_TIME)])  # EVENT_TIME is UTC-aware
        events_df = pd.DataFrame([{"id": 1, "time": naive_time}])

        result = match_images_to_events(images_df, events_df)
        assert len(result) == 1
        assert result.iloc[0]["matched_images"] == ["a.jpg"]

    def test_empty_events_df_returns_empty_result_without_error(self, boundary_images_df):
        empty_events = _events([])
        result = match_images_to_events(boundary_images_df, empty_events)

        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_empty_images_df_returns_empty_result(self, single_event_df):
        empty_images = _images([])
        result = match_images_to_events(empty_images, single_event_df)
        assert result.empty

    def test_image_count_matches_length_of_matched_images(self, boundary_images_df, single_event_df):
        result = match_images_to_events(boundary_images_df, single_event_df)
        row = result.iloc[0]
        assert row["image_count"] == len(row["matched_images"])

    def test_duplicate_image_datetimes_all_matched(self, single_event_df):
        images_df = _images(
            [
                ("dup1.jpg", EVENT_TIME),
                ("dup2.jpg", EVENT_TIME),
            ]
        )
        result = match_images_to_events(images_df, single_event_df)
        assert sorted(result.iloc[0]["matched_images"]) == ["dup1.jpg", "dup2.jpg"]

    def test_tz_naive_images_datetime_raises_type_error(self, single_event_df):
        # Suspected bug: `match_images_to_events` localizes a naive
        # `event["time"]` to UTC, but never coerces `images_df["datetime"]`.
        # If the images datetime column is tz-naive, comparing it against the
        # (now tz-aware) event_time raises, rather than being handled
        # gracefully or documented as a precondition.
        naive_images_df = _images([("a.jpg", pd.Timestamp("2024-06-01 10:00:00"))])  # no tz
        with pytest.raises(TypeError):
            match_images_to_events(naive_images_df, single_event_df)


class TestGetUnmatchedImages:
    def test_returns_images_not_present_in_matched_paths(self, boundary_images_df):
        matched_df = pd.DataFrame({"matched_images": [["lower.jpg", "ontime.jpg", "upper.jpg"]]})
        result = get_unmatched_images(boundary_images_df, matched_df)

        assert sorted(result["file_path"].tolist()) == ["early.jpg", "late.jpg"]

    def test_excludes_images_without_datetime(self, boundary_images_df):
        matched_df = pd.DataFrame({"matched_images": [[]]})
        result = get_unmatched_images(boundary_images_df, matched_df)

        assert "no_dt.jpg" not in result["file_path"].tolist()

    def test_all_images_matched_returns_empty(self, boundary_images_df):
        all_timestamped = boundary_images_df.dropna(subset=["datetime"])["file_path"].tolist()
        matched_df = pd.DataFrame({"matched_images": [all_timestamped]})

        result = get_unmatched_images(boundary_images_df, matched_df)
        assert result.empty

    def test_matched_df_with_multiple_rows_unions_paths(self, boundary_images_df):
        matched_df = pd.DataFrame({"matched_images": [["lower.jpg"], ["ontime.jpg", "upper.jpg"]]})
        result = get_unmatched_images(boundary_images_df, matched_df)
        assert sorted(result["file_path"].tolist()) == ["early.jpg", "late.jpg"]

    def test_matched_df_with_no_rows_but_correct_columns_returns_all_timestamped(self, boundary_images_df):
        matched_df = pd.DataFrame(columns=["matched_images"])
        result = get_unmatched_images(boundary_images_df, matched_df)

        expected = boundary_images_df.dropna(subset=["datetime"])["file_path"].tolist()
        assert sorted(result["file_path"].tolist()) == sorted(expected)

    def test_index_is_reset(self, boundary_images_df):
        matched_df = pd.DataFrame({"matched_images": [["lower.jpg"]]})
        result = get_unmatched_images(boundary_images_df, matched_df)
        assert list(result.index) == list(range(len(result)))

    def test_column_less_matched_df_raises_key_error(self, boundary_images_df):
        # Suspected bug: `match_images_to_events` returns a completely
        # column-less DataFrame (`pd.DataFrame(rows)` with `rows == []`) when
        # no event matches any image. Feeding that directly into
        # `get_unmatched_images` -- exactly as the docstring says it's meant
        # to be used -- raises a KeyError on "matched_images" instead of
        # being treated as "nothing matched".
        empty_matched_df = pd.DataFrame(match_images_to_events(boundary_images_df, _events([])))
        assert empty_matched_df.empty
        assert "matched_images" not in empty_matched_df.columns

        with pytest.raises(KeyError):
            get_unmatched_images(boundary_images_df, empty_matched_df)
