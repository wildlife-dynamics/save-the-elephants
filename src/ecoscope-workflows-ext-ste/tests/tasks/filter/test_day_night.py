"""Tests for ecoscope_workflows_ext_ste.tasks.filter._day_night.

`aggregate_day_night_fixes` is registered via `wt_registry.register()`, which
is a no-op at call time, so it behaves as a plain Python function here.

Contract (from source): group rows of `gdf` by `geometry`, count how many of
each group have `is_night` True/False, rename those counts to
`day_count_fixes` / `night_count_fixes`, and label each geometry `"Night"` if
night_count_fixes > day_count_fixes, else `"Day"` (so ties resolve to "Day").
The CRS of the input is preserved on the output GeoDataFrame.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from ecoscope_workflows_ext_ste.tasks.filter._day_night import (
    aggregate_day_night_fixes,
)


def _gdf(records, crs="EPSG:4326"):
    """records: list of (x, y, is_night) tuples."""
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(x, y) for x, y, _ in records],
            "is_night": [n for _, _, n in records],
        },
        crs=crs,
    )


class TestHappyPathMixedGeometries:
    def test_two_geometries_with_mixed_day_night_counts(self):
        gdf = _gdf(
            [
                (0, 0, True),
                (0, 0, True),
                (0, 0, False),
                (1, 1, False),
                (1, 1, False),
            ]
        )
        result = aggregate_day_night_fixes(gdf)

        assert len(result) == 2
        by_geom = {(pt.x, pt.y): row for pt, row in zip(result["geometry"], result.to_dict("records"))}

        origin = by_geom[(0.0, 0.0)]
        assert origin["day_count_fixes"] == 1
        assert origin["night_count_fixes"] == 2
        assert origin["dominant"] == "Night"

        other = by_geom[(1.0, 1.0)]
        assert other["day_count_fixes"] == 2
        assert other["night_count_fixes"] == 0
        assert other["dominant"] == "Day"

    def test_output_is_geodataframe_with_expected_columns_and_crs(self):
        gdf = _gdf([(0, 0, True), (1, 1, False)])
        result = aggregate_day_night_fixes(gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert set(result.columns) == {"geometry", "day_count_fixes", "night_count_fixes", "dominant"}
        assert result.crs == gdf.crs


class TestAllDayOrAllNight:
    def test_all_day_fixes_yields_zero_night_count_and_day_dominant(self):
        gdf = _gdf([(0, 0, False), (0, 0, False)])
        result = aggregate_day_night_fixes(gdf)

        assert result.loc[0, "day_count_fixes"] == 2
        assert result.loc[0, "night_count_fixes"] == 0
        assert result.loc[0, "dominant"] == "Day"

    def test_all_night_fixes_yields_zero_day_count_and_night_dominant(self):
        gdf = _gdf([(0, 0, True), (0, 0, True)])
        result = aggregate_day_night_fixes(gdf)

        assert result.loc[0, "day_count_fixes"] == 0
        assert result.loc[0, "night_count_fixes"] == 2
        assert result.loc[0, "dominant"] == "Night"


class TestTieBreaksToDay:
    def test_equal_day_and_night_counts_resolve_to_day(self):
        # dominant uses `night_count_fixes > day_count_fixes`, so a tie is
        # NOT dominant-night; it falls through to "Day".
        gdf = _gdf([(0, 0, True), (0, 0, False)])
        result = aggregate_day_night_fixes(gdf)

        assert result.loc[0, "day_count_fixes"] == 1
        assert result.loc[0, "night_count_fixes"] == 1
        assert result.loc[0, "dominant"] == "Day"


class TestMultipleDistinctGeometries:
    def test_three_distinct_geometries_each_get_own_row(self):
        gdf = _gdf(
            [
                (0, 0, True),
                (1, 1, False),
                (1, 1, False),
                (2, 2, True),
                (2, 2, False),
            ]
        )
        result = aggregate_day_night_fixes(gdf)
        assert len(result) == 3
        # every original point should appear exactly once in the aggregated output
        assert set(result["geometry"].apply(lambda p: (p.x, p.y))) == {(0, 0), (1, 1), (2, 2)}


class TestEmptyDataframe:
    def test_empty_input_returns_empty_geodataframe_with_expected_columns(self):
        gdf = gpd.GeoDataFrame({"geometry": [], "is_night": []}, crs="EPSG:4326")
        result = aggregate_day_night_fixes(gdf)

        assert result.empty
        assert set(result.columns) == {"geometry", "day_count_fixes", "night_count_fixes", "dominant"}


class TestMissingColumn:
    def test_missing_is_night_column_raises_key_error(self):
        gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
        with pytest.raises(KeyError):
            aggregate_day_night_fixes(gdf)

    def test_missing_geometry_column_raises(self):
        # A plain (non-geo) frame with no "geometry" column should fail the groupby.
        df = pd.DataFrame({"is_night": [True, False]})
        with pytest.raises(KeyError):
            aggregate_day_night_fixes(df)
