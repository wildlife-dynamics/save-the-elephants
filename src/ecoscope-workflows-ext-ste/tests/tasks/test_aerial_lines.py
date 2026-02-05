import pytest
import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from ecoscope_workflows_ext_ste.tasks._aerial_lines import (
    validate_polygon_geometry,
    draw_survey_lines,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_trajs_gdf():
    """Fixture to load the sample trajectories GeoDataFrame."""
    return gpd.read_file(TEST_DATA_DIR / "sample_trajs.gpkg").to_crs("EPSG:4326")


@pytest.fixture
def simple_polygon_gdf():
    """Fixture for a simple polygon GeoDataFrame."""
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    return gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")


@pytest.fixture
def multipolygon_gdf():
    """Fixture for a MultiPolygon GeoDataFrame."""
    poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
    multipoly = MultiPolygon([poly1, poly2])
    return gpd.GeoDataFrame(geometry=[multipoly], crs="EPSG:4326")


@pytest.fixture
def mixed_polygon_gdf():
    """Fixture with both Polygon and MultiPolygon geometries."""
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    multipoly = MultiPolygon(
        [Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]), Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])]
    )
    return gpd.GeoDataFrame(geometry=[poly, multipoly], crs="EPSG:4326")


@pytest.fixture
def polygon_with_null_gdf():
    """Fixture with polygon and null geometries."""
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    return gpd.GeoDataFrame(geometry=[poly, None, poly], crs="EPSG:4326")


@pytest.fixture
def invalid_geometry_gdf():
    """Fixture with invalid geometry types (Point and LineString)."""
    point = Point(0, 0)
    line = LineString([(0, 0), (5, 5)])
    poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    return gpd.GeoDataFrame(geometry=[point, line, poly], crs="EPSG:4326")


class TestValidatePolygonGeometry:
    """Test cases for validate_polygon_geometry function."""

    def test_valid_polygon_geometry(self, simple_polygon_gdf):
        """Test that valid polygon geometry passes validation."""
        result = validate_polygon_geometry(simple_polygon_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.geometry.geom_type[0] == "Polygon"

    def test_valid_multipolygon_geometry(self, multipolygon_gdf):
        """Test that valid MultiPolygon geometry passes validation."""
        result = validate_polygon_geometry(multipolygon_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result.geometry.geom_type[0] == "MultiPolygon"

    def test_mixed_valid_geometries(self, mixed_polygon_gdf):
        """Test that mixed Polygon and MultiPolygon geometries pass validation."""
        result = validate_polygon_geometry(mixed_polygon_gdf)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert all(gt in ["Polygon", "MultiPolygon"] for gt in result.geometry.geom_type)

    def test_null_geometry_removed(self, polygon_with_null_gdf):
        """Test that null geometries are removed during validation."""
        result = validate_polygon_geometry(polygon_with_null_gdf)
        assert len(result) == 2  # Two valid polygons, null removed
        assert result.geometry.isna().sum() == 0

    def test_invalid_geometry_raises_error(self, invalid_geometry_gdf):
        """Test that invalid geometry types raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_polygon_geometry(invalid_geometry_gdf)

        assert "Invalid geometry types found" in str(exc_info.value)
        assert "Point" in str(exc_info.value) or "LineString" in str(exc_info.value)


class TestDrawSurveyLines:
    """Test cases for draw_survey_lines function."""

    def test_north_south_lines(self, simple_polygon_gdf):
        """Test generation of North-South survey lines."""
        result = draw_survey_lines(simple_polygon_gdf, direction="North South", spacing=500)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert all(result.geometry.geom_type == "LineString")

        # Check that lines are roughly vertical (North-South)
        for geom in result.geometry:
            coords = list(geom.coords)
            assert len(coords) >= 2
            # X coordinates should be similar (vertical line)
            x_coords = [c[0] for c in coords]
            assert np.std(x_coords) < 100  # Small variance in X

    def test_east_west_lines(self, simple_polygon_gdf):
        """Test generation of East-West survey lines."""
        result = draw_survey_lines(simple_polygon_gdf, direction="East West", spacing=500)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert all(result.geometry.geom_type == "LineString")

        # Check that lines are roughly horizontal (East-West)
        for geom in result.geometry:
            coords = list(geom.coords)
            assert len(coords) >= 2
            # Y coordinates should be similar (horizontal line)
            y_coords = [c[1] for c in coords]
            assert np.std(y_coords) < 100  # Small variance in Y

    def test_spacing_parameter(self, simple_polygon_gdf):
        """Test that spacing parameter affects number of lines generated."""
        result_500 = draw_survey_lines(simple_polygon_gdf, spacing=500)
        result_1000 = draw_survey_lines(simple_polygon_gdf, spacing=1000)

        # Larger spacing should produce fewer lines
        assert len(result_500) > len(result_1000)

    def test_multipolygon_exploded(self, multipolygon_gdf):
        """Test that MultiPolygon geometries are properly handled."""
        result = draw_survey_lines(multipolygon_gdf, direction="North South", spacing=500)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        # Lines should be clipped to polygon boundaries
        assert all(result.geometry.geom_type == "LineString")

    def test_crs_conversion(self, simple_polygon_gdf):
        """Test that CRS is properly converted to EPSG:3857."""
        original_crs = simple_polygon_gdf.crs
        result = draw_survey_lines(simple_polygon_gdf, spacing=500)

        # Input should remain unchanged
        assert simple_polygon_gdf.crs == original_crs
        # Result should be in EPSG:3857
        assert result.crs.to_epsg() == 3857

    def test_invalid_direction_raises_error(self, simple_polygon_gdf):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            draw_survey_lines(simple_polygon_gdf, direction="Diagonal", spacing=500)

        assert "Direction must be 'North South' or 'East West'" in str(exc_info.value)

    def test_lines_clipped_to_polygon(self, simple_polygon_gdf):
        """Test that survey lines are clipped to polygon boundaries."""
