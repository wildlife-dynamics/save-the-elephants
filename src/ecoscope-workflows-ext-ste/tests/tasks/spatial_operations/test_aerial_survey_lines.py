"""Tests for ecoscope_workflows_ext_ste.tasks.spatial_operations._aerial_survey_lines.

Both `ensure_polygon_type` and `create_survey_transects` are registered via
`wt_registry.register()`, which is a no-op at call time, so they behave as
plain Python functions here.

Note on behavior vs. the pre-reorg test suite: the old test suite (for the
predecessor `validate_polygon_geometry` function) expected null geometries to
be silently dropped. The current `ensure_polygon_type` does *not* drop nulls:
`gdf.geometry.geom_type` reports `None` for a null geometry, and `None` is not
in the allowed {"Polygon", "MultiPolygon"} set, so a GeoDataFrame containing a
null geometry now raises `ValueError` instead of having the null silently
removed. This was verified empirically against the current source and is
treated as the correct current contract below (not "fixed").
"""

from pathlib import Path

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from ecoscope_workflows_ext_ste.tasks.spatial_operations._aerial_survey_lines import (
    create_survey_transects,
    ensure_polygon_type,
)

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def simple_polygon_gdf():
    """A small (~11km) square near the equator/prime meridian, in EPSG:4326."""
    polygon = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


@pytest.fixture
def multipolygon_gdf():
    poly1 = Polygon([(0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)])
    poly2 = Polygon([(0.1, 0.1), (0.15, 0.1), (0.15, 0.15), (0.1, 0.15)])
    return gpd.GeoDataFrame(geometry=[MultiPolygon([poly1, poly2])], crs="EPSG:4326")


@pytest.fixture
def mixed_polygon_gdf():
    poly = Polygon([(0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)])
    multipoly = MultiPolygon(
        [
            Polygon([(0.1, 0.1), (0.15, 0.1), (0.15, 0.15), (0.1, 0.15)]),
            Polygon([(0.2, 0.2), (0.25, 0.2), (0.25, 0.25), (0.2, 0.25)]),
        ]
    )
    return gpd.GeoDataFrame(geometry=[poly, multipoly], crs="EPSG:4326")


@pytest.fixture
def polygon_with_null_gdf():
    poly = Polygon([(0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)])
    return gpd.GeoDataFrame(geometry=[poly, None, poly], crs="EPSG:4326")


@pytest.fixture
def invalid_geometry_gdf():
    point = Point(0, 0)
    line = LineString([(0, 0), (0.05, 0.05)])
    poly = Polygon([(0, 0), (0.05, 0), (0.05, 0.05), (0, 0.05)])
    return gpd.GeoDataFrame(geometry=[point, line, poly], crs="EPSG:4326")


@pytest.fixture
def aoi_gdf():
    """Real-world AOI polygon fixture (single MultiPolygon feature)."""
    aois = gpd.read_file(TEST_DATA_DIR / "AOIs.gpkg", layer="AOIs")
    return aois.iloc[[0]].reset_index(drop=True)


class TestEnsurePolygonType:
    def test_valid_polygon_passes_through_unchanged(self, simple_polygon_gdf):
        result = ensure_polygon_type(simple_polygon_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.geometry.geom_type.iloc[0] == "Polygon"

    def test_valid_multipolygon_passes(self, multipolygon_gdf):
        result = ensure_polygon_type(multipolygon_gdf)
        assert len(result) == 1
        assert result.geometry.geom_type.iloc[0] == "MultiPolygon"

    def test_mixed_polygon_and_multipolygon_passes(self, mixed_polygon_gdf):
        result = ensure_polygon_type(mixed_polygon_gdf)
        assert len(result) == 2
        assert set(result.geometry.geom_type) == {"Polygon", "MultiPolygon"}

    def test_empty_geodataframe_passes(self):
        empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        result = ensure_polygon_type(empty)
        assert len(result) == 0

    def test_null_geometry_raises_value_error(self, polygon_with_null_gdf):
        # Current behavior: null geometries are NOT dropped; they surface as
        # an invalid "geometry type" (None) and raise. See module docstring.
        with pytest.raises(ValueError, match="Invalid geometry types"):
            ensure_polygon_type(polygon_with_null_gdf)

    def test_invalid_geometry_types_raises_value_error(self, invalid_geometry_gdf):
        with pytest.raises(ValueError) as exc_info:
            ensure_polygon_type(invalid_geometry_gdf)
        message = str(exc_info.value)
        assert "Invalid geometry types" in message
        assert "Point" in message
        assert "LineString" in message

    def test_returns_same_object_reference(self, simple_polygon_gdf):
        # ensure_polygon_type validates and returns the same gdf (no copy).
        result = ensure_polygon_type(simple_polygon_gdf)
        assert result is simple_polygon_gdf


class TestCreateSurveyTransects:
    def test_north_south_lines_are_vertical(self, simple_polygon_gdf):
        result = create_survey_transects(simple_polygon_gdf, direction="North South", spacing=2000)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert all(result.geometry.geom_type == "LineString")

        for geom in result.geometry:
            coords = list(geom.coords)
            assert len(coords) >= 2
            x_coords = [c[0] for c in coords]
            # Vertical line: x should barely vary (degrees, near-equal meridian).
            assert np.std(x_coords) < 1e-3

    def test_east_west_lines_are_horizontal(self, simple_polygon_gdf):
        result = create_survey_transects(simple_polygon_gdf, direction="East West", spacing=2000)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert all(result.geometry.geom_type == "LineString")

        for geom in result.geometry:
            coords = list(geom.coords)
            y_coords = [c[1] for c in coords]
            assert np.std(y_coords) < 1e-3

    def test_larger_spacing_produces_fewer_lines(self, simple_polygon_gdf):
        result_small_spacing = create_survey_transects(simple_polygon_gdf, spacing=1000)
        result_large_spacing = create_survey_transects(simple_polygon_gdf, spacing=5000)

        assert len(result_small_spacing) > len(result_large_spacing)

    def test_lines_clipped_within_polygon_bounds(self, simple_polygon_gdf):
        polygon = simple_polygon_gdf.geometry.iloc[0]
        result = create_survey_transects(simple_polygon_gdf, direction="North South", spacing=2000)

        buffered = polygon.buffer(1e-9)
        assert all(buffered.contains(geom) or buffered.intersects(geom) for geom in result.geometry)

        minx, miny, maxx, maxy = result.total_bounds
        pminx, pminy, pmaxx, pmaxy = polygon.bounds
        assert minx >= pminx - 1e-6
        assert maxx <= pmaxx + 1e-6
        assert miny >= pminy - 1e-6
        assert maxy <= pmaxy + 1e-6

    def test_multipolygon_input_produces_lines(self, multipolygon_gdf):
        result = create_survey_transects(multipolygon_gdf, direction="North South", spacing=2000)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert set(result.geometry.geom_type) <= {"LineString", "MultiLineString"}

    def test_output_crs_matches_input_crs(self, simple_polygon_gdf):
        original_crs = simple_polygon_gdf.crs
        result = create_survey_transects(simple_polygon_gdf, spacing=2000)

        # Input is not mutated in place.
        assert simple_polygon_gdf.crs == original_crs
        # Output is reprojected back to the original CRS.
        assert result.crs == original_crs

    def test_explicit_planar_crs_used_for_spacing(self, simple_polygon_gdf):
        result = create_survey_transects(simple_polygon_gdf, planar_crs="ESRI:102022", spacing=2000)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert result.crs == simple_polygon_gdf.crs

    def test_invalid_direction_raises_value_error(self, simple_polygon_gdf):
        with pytest.raises(ValueError, match="direction must be 'North South' or 'East West'"):
            create_survey_transects(simple_polygon_gdf, direction="Diagonal", spacing=2000)

    def test_missing_crs_raises_value_error(self, simple_polygon_gdf):
        no_crs_gdf = simple_polygon_gdf.copy()
        no_crs_gdf.crs = None
        with pytest.raises(ValueError, match="Input GeoDataFrame must have a CRS set"):
            create_survey_transects(no_crs_gdf, spacing=2000)

    def test_real_aoi_fixture_produces_clipped_lines(self, aoi_gdf):
        result = create_survey_transects(aoi_gdf, direction="North South", spacing=2000)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert set(result.geometry.geom_type) <= {"LineString", "MultiLineString"}
        assert result.crs == aoi_gdf.crs

        aoi_geom = aoi_gdf.geometry.iloc[0]
        buffered = aoi_geom.buffer(1e-9)
        assert all(buffered.intersects(geom) for geom in result.geometry)
