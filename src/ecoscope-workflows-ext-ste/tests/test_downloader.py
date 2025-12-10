import pytest
import geopandas as gpd
from shapely.geometry import Polygon, Point, LineString
from ecoscope_workflows_ext_ste.tasks._aerial_survey import (
    DownloadFile,
    LocalFile,
    validate_polygon_geometry,
    generate_survey_lines,
)


def test_download_file_creation():
    url = "https://www.dropbox.com/s/nvdmidz1o2duyl3/AOIs.gpkg?dl=1"
    download_file = DownloadFile(url=url)
    assert download_file.url == url


def test_local_file_creation():
    file_path = "AOIs.gpkg"
    local_file = LocalFile(file_path=file_path)
    assert local_file.file_path == file_path


def test_validate_polygon_geometry_valid():
    """Test validate_polygon_geometry with valid Polygon geometries"""
    gdf = gpd.GeoDataFrame(
        {"name": ["area1", "area2"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
    )

    result = validate_polygon_geometry(gdf)

    assert len(result) == 2
    assert all(result.geometry.geom_type == "Polygon")


def test_validate_polygon_geometry_invalid():
    """Test validate_polygon_geometry raises error with invalid geometry types"""
    gdf = gpd.GeoDataFrame({"name": ["point1", "line1"]}, geometry=[Point(0, 0), LineString([(0, 0), (1, 1)])])

    with pytest.raises(ValueError, match="Invalid geometry types found"):
        validate_polygon_geometry(gdf)


def test_generate_survey_lines_north_south():
    """Test generate_survey_lines creates vertical lines"""
    gdf = gpd.GeoDataFrame(
        {"name": ["area1"]}, geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])], crs="EPSG:3857"
    )

    result = generate_survey_lines(gdf, direction="North South", spacing=500)

    assert len(result) > 0
    assert result.crs == gdf.crs
    assert all(geom.geom_type == "LineString" for geom in result.geometry)


def test_generate_survey_lines_east_west():
    """Test generate_survey_lines creates horizontal lines"""
    gdf = gpd.GeoDataFrame(
        {"name": ["area1"]}, geometry=[Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])], crs="EPSG:3857"
    )

    result = generate_survey_lines(gdf, direction="East West", spacing=500)

    assert len(result) > 0
    assert result.crs == gdf.crs
    assert all(geom.geom_type == "LineString" for geom in result.geometry)


def test_generate_survey_lines_reprojects_wgs84():
    """Test generate_survey_lines reprojects WGS84 to Web Mercator"""
    gdf = gpd.GeoDataFrame(
        {"name": ["area1"]}, geometry=[Polygon([(36.0, -1.0), (37.0, -1.0), (37.0, 0.0), (36.0, 0.0)])], crs="EPSG:4326"
    )

    result = generate_survey_lines(gdf, direction="North South", spacing=10000)

    assert result.crs.to_string() == "EPSG:3857"
    assert len(result) > 0
