"""Tests for ecoscope_workflows_ext_ste.tasks.filter._filter.

`filter_rows` is registered via `wt_registry.register()`, which is a no-op at
call time, so it behaves as a plain Python function here -- the
`Annotated[...]` signature is not enforced by pydantic at runtime, only the
`match op:` branch logic and `reset_index` behavior matter.

Note: `filter_rows.reset_index` calls `.reset_index(drop=True)`, which drops
the old index entirely (unlike the sibling `filter_df` task in
`ecoscope.platform.tasks.transformation._filter`, which calls plain
`.reset_index()` and keeps the old index as a column). This is intentional
per this module and is exercised below.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope.platform.tasks.transformation._filter import ComparisonOperator
from ecoscope_workflows_ext_ste.tasks.filter._filter import filter_rows


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=[10, 11, 12, 13, 14])


@pytest.fixture
def string_df() -> pd.DataFrame:
    return pd.DataFrame({"name": ["alpha", "bravo", "charlie"]})


class TestEachComparisonOperator:
    def test_equal(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.EQUAL, 3)
        assert result["a"].tolist() == [3]

    def test_ne(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.NE, 3)
        assert result["a"].tolist() == [1, 2, 4, 5]

    def test_ge(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GE, 3)
        assert result["a"].tolist() == [3, 4, 5]

    def test_gt(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GT, 3)
        assert result["a"].tolist() == [4, 5]

    def test_le(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.LE, 3)
        assert result["a"].tolist() == [1, 2, 3]

    def test_lt(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.LT, 3)
        assert result["a"].tolist() == [1, 2]


class TestStringColumn:
    def test_equal_on_string_value(self, string_df):
        result = filter_rows(string_df, "name", ComparisonOperator.EQUAL, "bravo")
        assert result["name"].tolist() == ["bravo"]

    def test_ne_on_string_value(self, string_df):
        result = filter_rows(string_df, "name", ComparisonOperator.NE, "bravo")
        assert result["name"].tolist() == ["alpha", "charlie"]

    def test_lexicographic_gt_on_string_value(self, string_df):
        result = filter_rows(string_df, "name", ComparisonOperator.GT, "bravo")
        assert result["name"].tolist() == ["charlie"]


class TestBooleanColumn:
    def test_equal_true(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = filter_rows(df, "flag", ComparisonOperator.EQUAL, True)
        assert result["flag"].tolist() == [True, True]

    def test_ne_false(self):
        df = pd.DataFrame({"flag": [True, False, True]})
        result = filter_rows(df, "flag", ComparisonOperator.NE, False)
        assert result["flag"].tolist() == [True, True]


class TestResetIndex:
    def test_reset_index_false_by_default_preserves_original_index(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GE, 3)
        assert result.index.tolist() == [12, 13, 14]

    def test_reset_index_true_drops_old_index_and_renumbers(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GE, 3, reset_index=True)
        assert result.index.tolist() == [0, 1, 2]
        # drop=True means the old index values are NOT retained as a column
        assert "index" not in result.columns
        assert result["a"].tolist() == [3, 4, 5]

    def test_reset_index_true_on_empty_result(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GT, 999, reset_index=True)
        assert result.empty
        assert result.index.tolist() == []


class TestEmptyDataframe:
    def test_filtering_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame({"a": []})
        result = filter_rows(empty, "a", ComparisonOperator.EQUAL, 3)
        assert result.empty
        assert list(result.columns) == ["a"]


class TestNoMatchingRows:
    def test_no_rows_match_returns_empty_but_preserves_columns(self, numeric_df):
        result = filter_rows(numeric_df, "a", ComparisonOperator.GT, 1000)
        assert result.empty
        assert list(result.columns) == ["a"]


class TestMissingColumn:
    def test_missing_column_raises_key_error(self, numeric_df):
        with pytest.raises(KeyError):
            filter_rows(numeric_df, "does_not_exist", ComparisonOperator.EQUAL, 3)


class TestGeoDataFrameSupport:
    def test_filters_rows_of_a_geodataframe_and_keeps_geometry_and_crs(self):
        gdf = gpd.GeoDataFrame(
            {"category": ["a", "b", "a"], "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
            crs="EPSG:4326",
        )
        result = filter_rows(gdf, "category", ComparisonOperator.EQUAL, "a")
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.crs == gdf.crs
        assert [pt.x for pt in result["geometry"]] == [0, 2]


class TestDoesNotMutateInput:
    def test_original_dataframe_is_unchanged(self, numeric_df):
        original = numeric_df.copy()
        filter_rows(numeric_df, "a", ComparisonOperator.GE, 3, reset_index=True)
        pd.testing.assert_frame_equal(numeric_df, original)
