"""Tests for ecoscope_workflows_ext_ste.tasks.transformation._label.

`label_by_percentile_threshold` is registered via `wt_registry.register()`,
which is a no-op at call time, so it behaves as a plain Python function
here.

Behavior confirmed by reading the source:
    threshold = np.percentile(gdf[column].dropna(), pct)
    frac = pct / 100
    low_label  = f"0-{frac:g}"
    high_label = f"{frac:g}-1"
    gdf["label"] = np.where(gdf[column] >= threshold, high_label, low_label)
    gdf["threshold"] = threshold

Two behaviors worth pinning down explicitly (not "bugs" per se, but real
edge-case behavior a caller should know about):
  1. `dropna()` is only applied when *computing* the percentile threshold;
     the subsequent `np.where(gdf[column] >= threshold, ...)` comparison
     runs over the *original* (non-dropped) column. Any NaN compared with
     `>=` evaluates to False in numpy, so rows with a NaN value in
     `column` always get the *low* label rather than, say, a null/label
     of their own.
  2. `np.percentile` on an all-NaN-dropped (i.e. effectively empty) array
     raises `IndexError` (confirmed empirically against this repo's numpy
     1.26.4: `IndexError: index -1 is out of bounds for axis 0 with size 0`),
     not a friendlier `ValueError`. Calling this task with an empty
     dataframe (or one where `column` is all-NaN) surfaces that raw
     IndexError.
"""

import numpy as np
import pandas as pd
import pytest

from ecoscope_workflows_ext_ste.tasks.transformation._label import (
    label_by_percentile_threshold,
)


class TestHappyPath:
    def test_default_column_and_percentile(self):
        gdf = pd.DataFrame({"density": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = label_by_percentile_threshold(gdf)
        expected_threshold = np.percentile(gdf["density"], 65)
        assert result["threshold"].iloc[0] == expected_threshold
        expected_labels = np.where(gdf["density"] >= expected_threshold, "0.65-1", "0-0.65")
        assert list(result["label"]) == list(expected_labels)

    def test_custom_column_name(self):
        gdf = pd.DataFrame({"value": [10, 20, 30, 40, 50]})
        result = label_by_percentile_threshold(gdf, column="value", pct=50)
        expected_threshold = np.percentile(gdf["value"], 50)
        assert result["threshold"].iloc[0] == expected_threshold
        assert set(result["label"]) <= {"0-0.5", "0.5-1"}

    def test_threshold_column_is_constant_across_rows(self):
        gdf = pd.DataFrame({"density": [1, 2, 3, 4, 5]})
        result = label_by_percentile_threshold(gdf, pct=40)
        assert result["threshold"].nunique() == 1

    def test_label_values_use_g_format_for_fraction(self):
        gdf = pd.DataFrame({"density": [1, 2, 3, 4]})
        result = label_by_percentile_threshold(gdf, pct=25)
        # frac = 0.25 -> "0-0.25" / "0.25-1"
        assert set(result["label"]) <= {"0-0.25", "0.25-1"}

    def test_does_not_mutate_input(self):
        gdf = pd.DataFrame({"density": [1, 2, 3]})
        original_columns = list(gdf.columns)
        label_by_percentile_threshold(gdf)
        assert list(gdf.columns) == original_columns

    def test_values_above_threshold_get_high_label(self):
        gdf = pd.DataFrame({"density": [0, 0, 0, 0, 100]})
        result = label_by_percentile_threshold(gdf, pct=50)
        # threshold at 50th percentile of [0,0,0,0,100] is 0 -> everything
        # >= 0 gets the high label.
        assert (result["label"] == "0.5-1").all()

    def test_values_below_threshold_get_low_label(self):
        gdf = pd.DataFrame({"density": [1, 2, 3, 4, 100]})
        result = label_by_percentile_threshold(gdf, pct=10)
        threshold = np.percentile(gdf["density"], 10)
        expected = np.where(gdf["density"] >= threshold, "0.1-1", "0-0.1")
        assert list(result["label"]) == list(expected)


class TestBoundaryPercentiles:
    def test_pct_zero_uses_minimum_as_threshold(self):
        gdf = pd.DataFrame({"density": [5, 10, 15]})
        result = label_by_percentile_threshold(gdf, pct=0)
        assert result["threshold"].iloc[0] == 5
        # frac=0 -> low_label "0-0", high_label "0-1"; every value >= min
        # so every row gets the high label.
        assert (result["label"] == "0-1").all()

    def test_pct_100_uses_maximum_as_threshold(self):
        gdf = pd.DataFrame({"density": [5, 10, 15]})
        result = label_by_percentile_threshold(gdf, pct=100)
        assert result["threshold"].iloc[0] == 15
        # frac=1 -> low_label "0-1", high_label "1-1"; only the max value
        # qualifies as >= threshold.
        assert result.loc[result["density"] == 15, "label"].iloc[0] == "1-1"
        assert (result.loc[result["density"] < 15, "label"] == "0-1").all()

    def test_single_row(self):
        gdf = pd.DataFrame({"density": [42]})
        result = label_by_percentile_threshold(gdf, pct=65)
        assert result["threshold"].iloc[0] == 42
        # the single value always equals its own percentile -> high label.
        assert result["label"].iloc[0] == "0.65-1"


class TestNaNHandling:
    def test_nan_excluded_from_threshold_computation(self):
        gdf = pd.DataFrame({"density": [1.0, 2.0, 3.0, np.nan]})
        result = label_by_percentile_threshold(gdf, pct=50)
        expected_threshold = np.percentile([1.0, 2.0, 3.0], 50)
        assert result["threshold"].iloc[0] == expected_threshold

    def test_nan_rows_receive_low_label_not_their_own_class(self):
        # np.nan >= threshold is always False in numpy, so NaN rows fall
        # into the "low" bucket even though they were excluded from the
        # threshold computation itself.
        gdf = pd.DataFrame({"density": [10.0, 20.0, np.nan]})
        result = label_by_percentile_threshold(gdf, pct=50)
        nan_row_label = result.loc[result["density"].isna(), "label"].iloc[0]
        assert nan_row_label == "0-0.5"

    def test_all_nan_column_raises_index_error(self):
        gdf = pd.DataFrame({"density": [np.nan, np.nan]})
        with pytest.raises(IndexError):
            label_by_percentile_threshold(gdf, pct=50)


class TestEmptyInput:
    def test_empty_dataframe_raises_index_error(self):
        gdf = pd.DataFrame({"density": pd.Series([], dtype="float64")})
        with pytest.raises(IndexError):
            label_by_percentile_threshold(gdf, pct=65)


class TestRealFixtureData:
    def test_labels_real_trajectory_speed_column(self, sample_trajs_gdf):
        result = label_by_percentile_threshold(sample_trajs_gdf, column="speed_kmhr", pct=65)
        expected_threshold = np.percentile(sample_trajs_gdf["speed_kmhr"].dropna(), 65)
        assert result["threshold"].iloc[0] == expected_threshold
        assert set(result["label"].unique()) <= {"0-0.65", "0.65-1"}
        assert len(result) == len(sample_trajs_gdf)
