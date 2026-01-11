import os
import pytest
from pathlib import Path
import geopandas as gpd
from ecoscope_workflows_ext_ste.tasks._path_utils import get_local_geo_path, validate_geo_file

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def kenyan_counties_gdf():
    """Fixture to load the Kenyan counties GeoDataFrame."""
    return gpd.read_file(TEST_DATA_DIR, "kenyan_counties.gpkg")


@pytest.fixture
def kenyan_counties_path():
    """Fixture to get the path to the Kenyan counties file."""
    return Path(TEST_DATA_DIR) / "kenyan_counties.gpkg"


def test_valid_gpkg_with_real_file(kenyan_counties_path):
    """Test validation with real kenyan_counties.gpkg file."""
    result = validate_geo_file(kenyan_counties_path)
    assert result == kenyan_counties_path


def test_valid_shapefile_extension(tmp_path):
    """Test that .shp files pass validation."""
    file_path = tmp_path / "test.shp"
    file_path.touch()

    result = validate_geo_file(file_path)
    assert result == file_path


def test_valid_geoparquet_extension(tmp_path):
    """Test that .geoparquet files pass validation."""
    file_path = tmp_path / "test.geoparquet"
    file_path.touch()

    result = validate_geo_file(file_path)
    assert result == file_path


def test_invalid_extension_raises_error(tmp_path):
    """Test that invalid file formats raise ValueError."""
    file_path = tmp_path / "test.txt"
    file_path.touch()

    with pytest.raises(ValueError, match="Invalid file format '.txt'"):
        validate_geo_file(file_path)


def test_case_insensitive_extension(tmp_path):
    """Test that uppercase extensions are valid."""
    file_path = tmp_path / "test.GPKG"
    file_path.touch()

    result = validate_geo_file(file_path)
    assert result == file_path


# get_local_geo_path


def test_get_path_for_kenyan_counties(kenyan_counties_path):
    """Test getting path for the real kenyan_counties.gpkg file."""
    result = get_local_geo_path(kenyan_counties_path)

    assert isinstance(result, str)
    assert result.endswith("kenyan_counties.gpkg")
    assert not result.startswith("file://")
    assert "data" in result  # Should contain the data directory


def test_get_path_for_shapefile(tmp_path):
    """Test getting path for a shapefile."""
    file_path = tmp_path / "test.shp"
    file_path.touch()

    result = get_local_geo_path(file_path)

    assert isinstance(result, str)
    assert result.endswith("test.shp")


def test_returns_normalized_string_path(kenyan_counties_path):
    """Test that returned path is a normalized string."""
    result = get_local_geo_path(kenyan_counties_path)

    assert isinstance(result, str)
    assert os.path.exists(result)  # Verify the file actually exists
