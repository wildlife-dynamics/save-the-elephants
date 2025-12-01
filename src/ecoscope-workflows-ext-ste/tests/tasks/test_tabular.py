# tests/test_tabular.py
import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from ecoscope_workflows_ext_ste.tasks import split_gdf_by_column, generate_mcp_gdf, round_off_values


@pytest.fixture
def sample_gdf():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "species": ["lion", "lion", "elephant", "zebra"], "value": [10, 20, 30, 40]})
    geometry = [Point(x, x) for x in range(len(df))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def test_split_gdf_by_column_normal(sample_gdf):
    res = split_gdf_by_column(sample_gdf, "species")
    assert isinstance(res, dict)
    assert set(res.keys()) == {"lion", "elephant", "zebra"}
    for key, g in res.items():
        assert isinstance(g, gpd.GeoDataFrame)
        assert all(g["species"] == key)


def test_split_gdf_by_column_empty_gdf():
    empty_gdf = gpd.GeoDataFrame(columns=["id", "species", "geometry"])
    with pytest.raises(ValueError, match="gdf is empty"):
        split_gdf_by_column(empty_gdf, "species")


def test_split_gdf_by_column_missing_column(sample_gdf):
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        split_gdf_by_column(sample_gdf, "nonexistent")


def test_generate_mcp_gdf_normal(sample_gdf):
    res = generate_mcp_gdf(sample_gdf)
    assert isinstance(res, gpd.GeoDataFrame)
    assert all(col in res.columns for col in ["area_m2", "area_km2", "mcp"])
    assert res.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]
    assert res.crs == sample_gdf.crs


def test_generate_mcp_gdf_empty_gdf():
    empty_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
    with pytest.raises(ValueError, match="gdf is empty"):
        generate_mcp_gdf(empty_gdf)


def test_generate_mcp_gdf_missing_crs(sample_gdf):
    gdf_no_crs = sample_gdf.copy()
    gdf_no_crs.crs = None
    with pytest.raises(ValueError, match="must have a CRS set"):
        generate_mcp_gdf(gdf_no_crs)


@pytest.mark.parametrize(
    "value,dp,expected",
    [
        (3.14159, 2, 3.14),
        (2.71828, 3, 2.718),
        (-1.2345, 1, -1.2),
        (0.0, 0, 0),
    ],
)
def test_round_off_values(value, dp, expected):
    assert round_off_values(value, dp) == expected


def test_generate_mcp_gdf_with_non_point_geometry():
    poly = Polygon([(0, 0), (1, 0), (1, 1)])
    gdf_poly = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")
    res = generate_mcp_gdf(gdf_poly)
    assert isinstance(res, gpd.GeoDataFrame)
    assert res.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon", "Point"]
