"""Tests for ecoscope_workflows_ext_ste.tasks.reporting._image_tables.

Covers the two `wt_registry.register()`-decorated functions
(`register()` is a no-op at call time, so these are exercised as ordinary
Python functions): `build_matched_table` and `build_unmatched_table`.

Both render a styled HTML snippet from a pandas DataFrame; these tests assert
on the structural/content substrings rather than doing a full HTML parse,
since exact whitespace/formatting isn't part of the function's contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecoscope_workflows_ext_ste.tasks.reporting._image_tables import (
    _CSS,
    build_matched_table,
    build_unmatched_table,
)


class TestBuildMatchedTable:
    def test_empty_dataframe_returns_fallback_message(self):
        result = build_matched_table(pd.DataFrame(columns=["serial_number", "event_time"]))
        assert "No images matched any events." in result
        assert result.startswith(_CSS)

    def test_single_row_renders_table_and_summary(self):
        df = pd.DataFrame(
            [
                {
                    "serial_number": 101,
                    "event_time": "2024-01-01 10:00:00",
                    "event_type": "carcass_report",
                    "event_type_display": "Carcass Report",
                    "image_count": 3,
                    "matched_images": ["/a/img1.jpg", "/a/img2.jpg", "/a/img3.jpg"],
                }
            ]
        )
        result = build_matched_table(df)

        assert "<table>" in result
        assert "101" in result
        assert "2024-01-01 10:00:00" in result
        assert "Carcass Report" in result  # display name used, not raw event_type
        assert "img1.jpg" in result and "img2.jpg" in result and "img3.jpg" in result
        assert "img1.jpg<br>img2.jpg<br>img3.jpg" in result
        assert "1 events matched" in result
        assert "3 images queued" in result

    def test_event_type_display_missing_column_falls_back_to_event_type(self):
        df = pd.DataFrame(
            [
                {
                    "serial_number": 1,
                    "event_time": "t",
                    "event_type": "raw_type",
                    "image_count": 1,
                    "matched_images": ["/a/x.jpg"],
                }
            ]
        )
        result = build_matched_table(df)
        assert "raw_type" in result

    def test_event_type_display_empty_string_falls_back_to_event_type(self):
        df = pd.DataFrame(
            [
                {
                    "serial_number": 1,
                    "event_time": "t",
                    "event_type": "raw_type",
                    "event_type_display": "",
                    "image_count": 1,
                    "matched_images": ["/a/x.jpg"],
                }
            ]
        )
        result = build_matched_table(df)
        assert "raw_type" in result

    def test_multiple_rows_preserve_order_and_sum_image_counts(self):
        df = pd.DataFrame(
            [
                {
                    "serial_number": 1,
                    "event_time": "t1",
                    "event_type": "a",
                    "event_type_display": "A",
                    "image_count": 2,
                    "matched_images": ["/x/1.jpg", "/x/2.jpg"],
                },
                {
                    "serial_number": 2,
                    "event_time": "t2",
                    "event_type": "b",
                    "event_type_display": "B",
                    "image_count": 4,
                    "matched_images": ["/y/1.jpg", "/y/2.jpg", "/y/3.jpg", "/y/4.jpg"],
                },
            ]
        )
        result = build_matched_table(df)
        assert result.index("t1") < result.index("t2")
        assert "2 events matched" in result
        assert "6 images queued" in result

    def test_filenames_use_basename_not_full_path(self):
        df = pd.DataFrame(
            [
                {
                    "serial_number": 1,
                    "event_time": "t",
                    "event_type": "a",
                    "event_type_display": "A",
                    "image_count": 1,
                    "matched_images": ["/deep/nested/path/photo.jpg"],
                }
            ]
        )
        result = build_matched_table(df)
        assert "photo.jpg" in result
        assert "/deep/nested/path/" not in result


class TestBuildUnmatchedTable:
    def test_empty_dataframe_returns_fallback_message(self):
        result = build_unmatched_table(pd.DataFrame(columns=["file_name", "datetime"]))
        assert "All images were matched to events." in result
        assert result.startswith(_CSS)

    def test_single_row_renders_table_and_summary(self):
        df = pd.DataFrame([{"file_name": "IMG_0001.jpg", "datetime": "2024-01-01 09:00:00"}])
        result = build_unmatched_table(df)

        assert "<table>" in result
        assert "IMG_0001.jpg" in result
        assert "2024-01-01 09:00:00" in result
        assert "1 images unmatched" in result

    def test_multiple_rows_all_present(self):
        df = pd.DataFrame(
            [
                {"file_name": "a.jpg", "datetime": "t1"},
                {"file_name": "b.jpg", "datetime": "t2"},
                {"file_name": "c.jpg", "datetime": "t3"},
            ]
        )
        result = build_unmatched_table(df)
        for name in ("a.jpg", "b.jpg", "c.jpg"):
            assert name in result
        assert "3 images unmatched" in result

    def test_nan_datetime_still_renders(self):
        df = pd.DataFrame([{"file_name": "a.jpg", "datetime": np.nan}])
        result = build_unmatched_table(df)
        assert "a.jpg" in result
        assert "1 images unmatched" in result
