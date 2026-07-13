"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._mcp.

`compute_minimum_convex_polygon` is registered via `wt_registry.register()`,
which is a no-op at call time, so it behaves as a plain Python function here.

Note on floating-point collinearity: reprojecting three exactly-collinear
points through a round trip (e.g. EPSG:4326 -> ESRI:102022 -> EPSG:4326 ->
ESRI:102022 again) does not generally preserve exact collinearity at double
precision, so `convex_hull` can come back as a degenerate sliver `Polygon`
rather than a `LineString`, and the function will *not* raise in that case.
To exercise the genuine "collinear points" validation branch deterministically,
the collinearity test below constructs points directly in the same CRS as
`planar_crs`, so the internal `to_crs` call is a no-op and exact collinearity
is preserved.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ecoscope_workflows_ext_ste.tasks.spatial_operations._mcp import (
    compute_minimum_convex_polygon,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"

PLANAR_CRS = "ESRI:102022"  # Africa Albers Equal Area (the function's default)


@pytest.fixture
def trajectory_gdf():
    """Real trajectory-segment fixture (LineString geometries)."""
    return gpd.read_file(TEST_DATA_DIR / "sample_trajs.gpkg")


@pytest.fixture
def triangle_points_gdf():
    return gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
        crs="EPSG:4326",
    )


class TestHappyPath:
    def test_real_trajectory_fixture_produces_valid_hull(self, trajectory_gdf):
        result = compute_minimum_convex_polygon(trajectory_gdf)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert set(result.columns) >= {"area_m2", "area_km2", "geometry"}
        assert result.crs == trajectory_gdf.crs

        geom = result.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        assert geom.is_valid
        assert geom.area > 0
        assert result["area_m2"].iloc[0] > 0
        assert result["area_km2"].iloc[0] == pytest.approx(result["area_m2"].iloc[0] / 1e6)

        # All input vertices should lie within (or on the boundary of) the hull.
        hull_buffered = geom.buffer(1e-9)
        all_points = [Point(c) for line in trajectory_gdf.geometry for c in line.coords]
        assert all(hull_buffered.contains(p) or hull_buffered.intersects(p) for p in all_points)

    def test_triangle_of_points_returns_expected_geometry(self, triangle_points_gdf):
        result = compute_minimum_convex_polygon(triangle_points_gdf, planar_crs=PLANAR_CRS)

        geom = result.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        assert geom.is_valid
        assert result["area_km2"].iloc[0] > 0

        for pt in triangle_points_gdf.geometry:
            assert geom.buffer(1e-9).contains(pt) or geom.buffer(1e-9).intersects(pt)

    def test_polygon_input_uses_all_vertices(self):
        # A square input polygon: the MCP of its vertices is itself (approximately).
        from shapely.geometry import Polygon

        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame(geometry=[square], crs="EPSG:4326")
        result = compute_minimum_convex_polygon(gdf, planar_crs=PLANAR_CRS)

        geom = result.geometry.iloc[0]
        assert geom.geom_type == "Polygon"
        assert geom.is_valid
        assert result["area_m2"].iloc[0] > 0

    def test_default_planar_crs_is_africa_albers(self, triangle_points_gdf):
        # Calling with and without an explicit (but identical) planar_crs should
        # produce the same area, confirming the documented default is used.
        result_default = compute_minimum_convex_polygon(triangle_points_gdf)
        result_explicit = compute_minimum_convex_polygon(triangle_points_gdf, planar_crs="ESRI:102022")
        assert result_default["area_m2"].iloc[0] == pytest.approx(result_explicit["area_m2"].iloc[0])

    def test_output_geometry_reprojected_to_input_crs(self, triangle_points_gdf):
        result = compute_minimum_convex_polygon(triangle_points_gdf, planar_crs=PLANAR_CRS)
        assert result.crs == triangle_points_gdf.crs
        assert result.crs.to_epsg() == 4326


class TestEdgeCasesTooFewPoints:
    def test_single_point_raises_value_error(self):
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="MCP requires at least 3 unique points"):
            compute_minimum_convex_polygon(gdf)

    def test_two_unique_points_raises_value_error(self):
        gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        with pytest.raises(ValueError, match="got 2"):
            compute_minimum_convex_polygon(gdf)

    def test_repeated_identical_points_counted_as_one(self):
        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(0, 0), Point(0, 0)],
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="got 1"):
            compute_minimum_convex_polygon(gdf)

    def test_exactly_collinear_points_raise_value_error(self):
        # Points constructed directly in the planar CRS so the internal
        # `to_crs(planar_crs)` call is a no-op and exact collinearity survives
        # (see module docstring for why a round-trip reprojection is unreliable
        # for this case).
        gdf = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1000, 1000), Point(2000, 2000)],
            crs=PLANAR_CRS,
        )
        with pytest.raises(ValueError, match="non-collinear"):
            compute_minimum_convex_polygon(gdf, planar_crs=PLANAR_CRS)


class TestValidationErrors:
    def test_missing_crs_raises_value_error(self, triangle_points_gdf):
        no_crs_gdf = triangle_points_gdf.copy()
        no_crs_gdf.crs = None
        with pytest.raises(ValueError, match="no CRS set"):
            compute_minimum_convex_polygon(no_crs_gdf)

    def test_empty_geodataframe_raises_value_error(self):
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty or contains no valid geometries"):
            compute_minimum_convex_polygon(empty)

    def test_all_null_geometries_raises_value_error(self):
        gdf = gpd.GeoDataFrame(geometry=[None, None, None], crs="EPSG:4326")
        with pytest.raises(ValueError, match="empty or contains no valid geometries"):
            compute_minimum_convex_polygon(gdf)

    def test_mixed_valid_and_null_geometries_uses_only_valid(self, triangle_points_gdf):
        gdf = gpd.GeoDataFrame(
            geometry=list(triangle_points_gdf.geometry) + [None],
            crs="EPSG:4326",
        )
        # Should not raise -- the null row is filtered out, leaving 3 valid points.
        result = compute_minimum_convex_polygon(gdf, planar_crs=PLANAR_CRS)
        assert result.geometry.iloc[0].geom_type == "Polygon"


class TestAreaCalculation:
    def test_larger_extent_yields_larger_area(self):
        small = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(0.01, 0), Point(0, 0.01)],
            crs="EPSG:4326",
        )
        large = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 0), Point(0, 1)],
            crs="EPSG:4326",
        )
        small_result = compute_minimum_convex_polygon(small, planar_crs=PLANAR_CRS)
        large_result = compute_minimum_convex_polygon(large, planar_crs=PLANAR_CRS)

        assert large_result["area_m2"].iloc[0] > small_result["area_m2"].iloc[0]
