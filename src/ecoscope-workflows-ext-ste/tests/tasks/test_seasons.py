import pytest
import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
from ecoscope_workflows_ext_ste.tasks._seasons import calculate_seasonal_home_range
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import CustomGridCellSize

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_gdf():
    """Fixture to load the sample GeoDataFrame."""
    return gpd.read_file(os.path.join(TEST_DATA_DIR, "sample_trajs.gpkg")).to_crs("EPSG:4326")


@pytest.fixture
def sample_seasons_gdf():
    """Fixture to load the sample GeoDataFrame."""
    return gpd.read_file(os.path.join(TEST_DATA_DIR, "sample_season_traj.gpkg")).to_crs("EPSG:4326")


@pytest.fixture
def sample_seasons_df():
    return pd.read_csv(os.path.join(TEST_DATA_DIR / "seasonal_windows.csv"))


def test_empty_gdf_raises_error():
    empty_gdf = gpd.GeoDataFrame(columns=["geometry", "season", "groupby_col"])
    with pytest.raises(ValueError, match="gdf is empty"):
        calculate_seasonal_home_range(empty_gdf)


def test_missing_season_column_raises_error(sample_seasons_gdf):
    gdf = sample_seasons_gdf.copy()
    gdf = gdf.drop(columns=["season"], errors="ignore")
    with pytest.raises(ValueError, match="season"):
        calculate_seasonal_home_range(gdf)


def test_auto_vs_custom_cell_size(sample_seasons_gdf):
    # Auto-scale (default)
    result_auto = calculate_seasonal_home_range(sample_seasons_gdf)
    assert isinstance(result_auto, pd.DataFrame)

    # Custom grid cell size
    custom_cell_size = CustomGridCellSize(cell_size=500)
    result_custom = calculate_seasonal_home_range(sample_seasons_gdf, auto_scale_or_custom_cell_size=custom_cell_size)
    assert isinstance(result_custom, pd.DataFrame)
