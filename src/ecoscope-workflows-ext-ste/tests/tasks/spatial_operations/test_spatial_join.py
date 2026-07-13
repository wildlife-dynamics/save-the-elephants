"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_join.

`spatial_join` is registered via `wt_registry.register()`, which is a no-op
at call time, so it behaves as a plain Python function here -- no pydantic
validation/coercion happens on the way in.

Ground truth for the real-data assertions below (AOIs.gpkg x kenya_pa.gpkg,
predicate="intersects") was independently verified with `geopandas.sjoin`
directly against the fixture files before being encoded as expectations:
  - how="inner": 39 matched rows.
  - how="left": 41 rows (39 matched + 2 AOIs with no PA overlap: "Rift /
    Mosiro" and "Mau Forest").
  - how="right": 124 rows (each `kenya_pa` row preserved at least once).
  - predicate="within": 0 rows (no AOI polygon is fully contained by a
    single protected-area polygon).
"""

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_join import (
    spatial_join,
)


class TestHappyPathRealFixtures:
    def test_inner_join_matches_expected_row_count(self, aois_gdf, kenya_pa_gdf):
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner", predicate="intersects")
        assert len(result) == 39

    def test_left_join_preserves_all_left_rows_at_least_once(self, aois_gdf, kenya_pa_gdf):
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="left", predicate="intersects")
        assert len(result) == 41
        # every original AOI name must appear at least once in the output
        assert set(aois_gdf["name"]) <= set(result["name_left"])

    def test_right_join_preserves_all_right_rows_at_least_once(self, aois_gdf, kenya_pa_gdf):
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="right", predicate="intersects")
        assert len(result) == 124
        assert set(kenya_pa_gdf["objectid"]) <= set(result["objectid"])

    def test_default_how_is_left(self, aois_gdf, kenya_pa_gdf):
        default_result = spatial_join(aois_gdf, kenya_pa_gdf)
        explicit_result = spatial_join(aois_gdf, kenya_pa_gdf, how="left", predicate="intersects")
        assert len(default_result) == len(explicit_result)

    def test_overlapping_non_geometry_columns_are_suffixed(self, aois_gdf, kenya_pa_gdf):
        # both frames have a "name" column -> collision must be resolved
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner")
        assert "name_left" in result.columns
        assert "name_right" in result.columns
        assert "name" not in result.columns

    def test_sjoin_internal_index_columns_are_stripped(self, aois_gdf, kenya_pa_gdf):
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner")
        assert "index_left" not in result.columns
        assert "index_right" not in result.columns

    def test_result_is_geodataframe_with_geometry_column(self, aois_gdf, kenya_pa_gdf):
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner")
        assert isinstance(result, gpd.GeoDataFrame)
        assert "geometry" in result.columns


class TestPredicateVariation:
    def test_within_predicate_yields_no_matches_for_broad_aois(self, aois_gdf, kenya_pa_gdf):
        # AOIs are broad regions, none is fully "within" a single PA polygon.
        result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner", predicate="within")
        assert len(result) == 0

    def test_within_is_stricter_than_intersects(self, aois_gdf, kenya_pa_gdf):
        intersects_result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner", predicate="intersects")
        within_result = spatial_join(aois_gdf, kenya_pa_gdf, how="inner", predicate="within")
        assert len(within_result) <= len(intersects_result)


class TestNonOverlappingGeometries:
    def test_inner_join_returns_empty_when_no_geometry_overlaps(self, aois_gdf, kenya_pa_gdf):
        rift = aois_gdf[aois_gdf["name"] == "Rift / Mosiro"].reset_index(drop=True)
        result = spatial_join(rift, kenya_pa_gdf, how="inner", predicate="intersects")
        assert len(result) == 0
        # columns are still suffixed even though nothing matched
        assert "name_left" in result.columns
        assert "name_right" in result.columns

    def test_left_join_keeps_unmatched_row_with_nan_right_columns(self, aois_gdf, kenya_pa_gdf):
        rift = aois_gdf[aois_gdf["name"] == "Rift / Mosiro"].reset_index(drop=True)
        result = spatial_join(rift, kenya_pa_gdf, how="left", predicate="intersects")
        assert len(result) == 1
        assert result["name_left"].iloc[0] == "Rift / Mosiro"
        assert result["objectid"].isna().all()

    def test_fully_disjoint_synthetic_polygons(self):
        left = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        right = gpd.GeoDataFrame(
            {"id": [2]},
            geometry=[Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])],
            crs="EPSG:4326",
        )
        result = spatial_join(left, right, how="inner")
        assert len(result) == 0


class TestCrsMismatch:
    def test_mismatched_crs_raises_value_error(self, aois_gdf, kenya_pa_gdf):
        reprojected = kenya_pa_gdf.to_crs("EPSG:32637")
        with pytest.raises(ValueError, match="CRS mismatch"):
            spatial_join(aois_gdf, reprojected)

    def test_error_message_includes_both_crs_values(self, aois_gdf, kenya_pa_gdf):
        reprojected = kenya_pa_gdf.to_crs("EPSG:32637")
        with pytest.raises(ValueError) as excinfo:
            spatial_join(aois_gdf, reprojected)
        message = str(excinfo.value)
        assert "4326" in message
        assert "32637" in message

    def test_missing_crs_on_right_df_raises(self, aois_gdf):
        no_crs = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        )
        assert no_crs.crs is None
        # left_df.crs.equals(None) is False -> ValueError before sjoin runs
        with pytest.raises(ValueError, match="CRS mismatch"):
            spatial_join(aois_gdf, no_crs)


class TestEmptyInputs:
    def test_empty_left_df_returns_empty_result(self, kenya_pa_gdf):
        empty_left = gpd.GeoDataFrame({"name": []}, geometry=gpd.GeoSeries([], dtype="geometry"), crs="EPSG:4326")
        result = spatial_join(empty_left, kenya_pa_gdf, how="inner")
        assert len(result) == 0

    def test_empty_right_df_with_left_join_keeps_left_rows(self, aois_gdf):
        empty_right = gpd.GeoDataFrame({"objectid": []}, geometry=gpd.GeoSeries([], dtype="geometry"), crs="EPSG:4326")
        result = spatial_join(aois_gdf, empty_right, how="left")
        assert len(result) == len(aois_gdf)
        assert result["objectid"].isna().all()
