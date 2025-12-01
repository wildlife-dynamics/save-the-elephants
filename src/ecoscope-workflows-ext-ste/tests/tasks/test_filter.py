import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point
from ecoscope_workflows_ext_ste.tasks import filter_by_value, exclude_by_value


@pytest.fixture
def sample_gdf():
    """Create a simple GeoDataFrame for testing."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "type": ["lion", "elephant", "lion", "zebra", "elephant"],
            "value": [10, 20, 30, 40, 50],
        }
    )
    geometry = [Point(x, x) for x in range(len(df))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def test_filter_by_value_single_value(sample_gdf):
    res = filter_by_value(sample_gdf, "type", "lion")
    # Expect only rows with type == "lion" (ids 1 and 3)
    assert len(res) == 2
    assert set(res["id"].tolist()) == {1, 3}


def test_filter_by_value_sequence(sample_gdf):
    res = filter_by_value(sample_gdf, "type", ["lion", "zebra"])
    # Expect lion and zebra -> ids 1,3,4
    assert len(res) == 3
    assert set(res["id"].tolist()) == {1, 3, 4}


def test_filter_by_value_not_present_returns_empty(sample_gdf):
    res = filter_by_value(sample_gdf, "type", "giraffe")
    assert res.shape[0] == 0


def test_exclude_by_value_single_value(sample_gdf):
    res = exclude_by_value(sample_gdf, "type", "elephant")
    # Exclude elephants -> remaining ids 1,3,4
    assert len(res) == 3
    assert set(res["id"].tolist()) == {1, 3, 4}


def test_exclude_by_value_sequence(sample_gdf):
    res = exclude_by_value(sample_gdf, "type", ["lion", "zebra"])
    # Exclude lion and zebra -> only elephants remain (ids 2,5)
    assert len(res) == 2
    assert set(res["id"].tolist()) == {2, 5}


def test_preserves_geometry_and_returns_copy(sample_gdf):
    # Ensure geometry column preserved for both operations and returned is a copy
    original_geom = sample_gdf.geometry.copy()

    res_filter = filter_by_value(sample_gdf, "type", "lion")
    # geometry column exists and is of GeoSeries type
    assert "geometry" in res_filter.columns
    assert isinstance(res_filter.geometry, gpd.GeoSeries)

    # Modify the returned gdf and ensure original isn't changed (copy semantics)
    res_filter.loc[:, "new_col"] = "x"
    assert "new_col" in res_filter.columns
    assert "new_col" not in sample_gdf.columns

    # Also check exclude_by_value
    res_exclude = exclude_by_value(sample_gdf, "type", "elephant")
    assert "geometry" in res_exclude.columns
    assert isinstance(res_exclude.geometry, gpd.GeoSeries)

    # Ensure original geometry still intact
    pd.testing.assert_series_equal(sample_gdf.geometry.reset_index(drop=True), original_geom.reset_index(drop=True))


def test_numeric_value_filtering():
    # Build a plain pandas DataFrame to verify numeric comparisons also work
    df = pd.DataFrame({"id": [1, 2, 3], "score": [0, 10, 20]})

    res = filter_by_value(df, "score", 10)
    assert len(res) == 1
    assert res.iloc[0]["id"] == 2

    res_ex = exclude_by_value(df, "score", [0, 20])
    # Only id 2 should remain
    assert len(res_ex) == 1
    assert res_ex.iloc[0]["id"] == 2
