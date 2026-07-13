"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_tag.

`spatial_tag` is registered via `wt_registry.register()`, which is a no-op
at call time, so it behaves as a plain Python function here.

Ground truth for the real-data assertions below (AOIs.gpkg tagged against
kenya_pa.gpkg, predicate="intersects") was independently verified with
geopandas directly against the fixture files:
  "Rift / Mosiro" and "Mau Forest" do not intersect any kenya_pa polygon;
  the other 5 AOIs ("Mara / Serengeti", "Marmanet Forest", "Loita Forest",
  "Samburu Reserve", "Nyakweri") each intersect at least one.
"""

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

from ecoscope_workflows_ext_ste.tasks.spatial_operations._spatial_tag import (
    spatial_tag,
)

UNMATCHED_AOIS = {"Rift / Mosiro", "Mau Forest"}
MATCHED_AOIS = {"Mara / Serengeti", "Marmanet Forest", "Loita Forest", "Samburu Reserve", "Nyakweri"}


class TestHappyPathRealFixturesMixedResult:
    def test_tags_known_unmatched_and_matched_aois_correctly(self, aois_gdf, kenya_pa_gdf):
        result = spatial_tag(aois_gdf, kenya_pa_gdf, output_column="pa_status")
        by_name = result.set_index("name")["pa_status"]
        for name in UNMATCHED_AOIS:
            assert by_name[name] == "Outside", name
        for name in MATCHED_AOIS:
            assert by_name[name] == "Inside", name

    def test_preserves_row_count_columns_and_crs(self, aois_gdf, kenya_pa_gdf):
        result = spatial_tag(aois_gdf, kenya_pa_gdf, output_column="pa_status")
        assert len(result) == len(aois_gdf)
        assert set(aois_gdf.columns) <= set(result.columns)
        assert result.crs == aois_gdf.crs

    def test_custom_matched_and_unmatched_labels(self, aois_gdf, kenya_pa_gdf):
        result = spatial_tag(
            aois_gdf,
            kenya_pa_gdf,
            output_column="protection_status",
            matched_label="Protected",
            unmatched_label="Unprotected",
        )
        by_name = result.set_index("name")["protection_status"]
        assert by_name["Rift / Mosiro"] == "Unprotected"
        assert by_name["Mara / Serengeti"] == "Protected"
        assert set(result["protection_status"].unique()) <= {"Protected", "Unprotected"}


class TestHappyPathAllMatched:
    def test_all_trajectory_segments_fall_inside_protected_areas(self, sample_trajs_gdf, kenya_pa_gdf):
        # Independently verified: 100% of sample_trajs.gpkg segments intersect
        # the union of kenya_pa.gpkg polygons (Samburu / Buffalo Springs area).
        result = spatial_tag(sample_trajs_gdf, kenya_pa_gdf, output_column="protection_status")
        assert (result["protection_status"] == "Inside").all()
        assert len(result) == len(sample_trajs_gdf)


class TestNoOverlap:
    def test_all_rows_tagged_unmatched_when_reference_is_disjoint(self, sample_trajs_gdf, aois_gdf):
        # "Rift / Mosiro" (~35.9-36.6E, -1.7 to 0.05N) is nowhere near the
        # Samburu trajectory data (~37.5-37.6E, 0.53-0.61N).
        rift = aois_gdf[aois_gdf["name"] == "Rift / Mosiro"].reset_index(drop=True)
        result = spatial_tag(sample_trajs_gdf, rift, output_column="status")
        assert (result["status"] == "Outside").all()


class TestEmptyReferenceGdf:
    def test_empty_reference_tags_everything_unmatched(self, aois_gdf, kenya_pa_gdf):
        empty_ref = kenya_pa_gdf.iloc[0:0]
        assert empty_ref.empty
        result = spatial_tag(aois_gdf, empty_ref, output_column="status")
        assert (result["status"] == "Outside").all()
        assert len(result) == len(aois_gdf)

    def test_empty_reference_respects_custom_unmatched_label(self, aois_gdf, kenya_pa_gdf):
        empty_ref = kenya_pa_gdf.iloc[0:0]
        result = spatial_tag(aois_gdf, empty_ref, output_column="status", unmatched_label="NoData")
        assert (result["status"] == "NoData").all()


class TestCrsReprojection:
    def test_reference_gdf_is_reprojected_to_match_df(self, aois_gdf, kenya_pa_gdf):
        reprojected_ref = kenya_pa_gdf.to_crs("EPSG:32637")
        assert not reprojected_ref.crs.equals(aois_gdf.crs)

        result = spatial_tag(aois_gdf, reprojected_ref, output_column="pa_status")
        expected = spatial_tag(aois_gdf, kenya_pa_gdf, output_column="pa_status")
        assert result["pa_status"].tolist() == expected["pa_status"].tolist()

    def test_output_crs_matches_df_crs_not_reference_crs(self, aois_gdf, kenya_pa_gdf):
        reprojected_ref = kenya_pa_gdf.to_crs("EPSG:32637")
        result = spatial_tag(aois_gdf, reprojected_ref, output_column="pa_status")
        assert result.crs == aois_gdf.crs


class TestMissingCrsValidation:
    def test_missing_crs_on_df_raises(self, kenya_pa_gdf):
        no_crs_df = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
        assert no_crs_df.crs is None
        with pytest.raises(ValueError, match=r"`df` must have a CRS set\."):
            spatial_tag(no_crs_df, kenya_pa_gdf, output_column="status")

    def test_missing_crs_on_reference_gdf_raises(self, aois_gdf):
        no_crs_ref = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])
        assert no_crs_ref.crs is None
        with pytest.raises(ValueError, match=r"`reference_gdf` must have a CRS set\."):
            spatial_tag(aois_gdf, no_crs_ref, output_column="status")


class TestPredicateVariation:
    """Deterministic synthetic geometries so each predicate's semantics are
    unambiguous, independent of the real fixture data."""

    @pytest.fixture
    def big_square(self) -> gpd.GeoDataFrame:
        # a 10x10 square centered at the origin
        return gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])],
            crs="EPSG:4326",
        )

    @pytest.fixture
    def small_square_inside(self) -> gpd.GeoDataFrame:
        # fully contained within big_square
        return gpd.GeoDataFrame(
            {"id": [2]},
            geometry=[Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])],
            crs="EPSG:4326",
        )

    def test_within_predicate_true_when_fully_contained(self, small_square_inside, big_square):
        result = spatial_tag(small_square_inside, big_square, output_column="status", predicate="within")
        assert (result["status"] == "Inside").all()

    def test_within_predicate_false_when_not_fully_contained(self, big_square, small_square_inside):
        # big_square is NOT within the smaller reference square
        result = spatial_tag(big_square, small_square_inside, output_column="status", predicate="within")
        assert (result["status"] == "Outside").all()

    def test_contains_predicate_true_when_df_geometry_contains_reference(self, big_square, small_square_inside):
        result = spatial_tag(big_square, small_square_inside, output_column="status", predicate="contains")
        assert (result["status"] == "Inside").all()

    def test_contains_predicate_false_when_reference_is_larger(self, small_square_inside, big_square):
        result = spatial_tag(small_square_inside, big_square, output_column="status", predicate="contains")
        assert (result["status"] == "Outside").all()

    def test_intersects_is_default_predicate(self, small_square_inside, big_square):
        default_result = spatial_tag(small_square_inside, big_square, output_column="status")
        explicit_result = spatial_tag(small_square_inside, big_square, output_column="status", predicate="intersects")
        assert default_result["status"].tolist() == explicit_result["status"].tolist()


class TestOutputColumnBehavior:
    def test_uses_numpy_where_producing_object_or_string_dtype(self, aois_gdf, kenya_pa_gdf):
        result = spatial_tag(aois_gdf, kenya_pa_gdf, output_column="status")
        values = set(result["status"].tolist())
        assert values <= {"Inside", "Outside"}
        assert isinstance(result["status"], type(result["status"]))  # sanity: still a Series

    def test_overwrites_existing_column_if_name_collides(self, aois_gdf, kenya_pa_gdf):
        aois_gdf["status"] = "placeholder"
        result = spatial_tag(aois_gdf, kenya_pa_gdf, output_column="status")
        assert set(result["status"].unique()) <= {"Inside", "Outside"}

    def test_does_not_mutate_input_df(self, aois_gdf, kenya_pa_gdf):
        original_columns = list(aois_gdf.columns)
        spatial_tag(aois_gdf, kenya_pa_gdf, output_column="status")
        assert list(aois_gdf.columns) == original_columns
