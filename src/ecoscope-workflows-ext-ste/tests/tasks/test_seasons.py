import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
from ecoscope_workflows_ext_ste.tasks._seasons import calculate_seasonal_home_range, create_seasonal_labels
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import CustomGridCellSize

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_gdf():
    """Fixture to load the sample GeoDataFrame."""
    return gpd.read_file(TEST_DATA_DIR, "sample_trajs.gpkg").to_crs("EPSG:4326")


@pytest.fixture
def sample_seasons_gdf():
    """Fixture to load the sample GeoDataFrame."""
    return gpd.read_file(TEST_DATA_DIR, "sample_season_traj.gpkg").to_crs("EPSG:4326")


@pytest.fixture
def sample_seasons_df():
    return pd.read_csv(TEST_DATA_DIR / "seasonal_windows.csv")


def test_empty_gdf_raises_error():
    empty_gdf = gpd.GeoDataFrame(columns=["geometry", "season", "groupby_col"])
    with pytest.raises(ValueError, match="gdf is empty"):
        calculate_seasonal_home_range(empty_gdf)


def test_missing_season_column_raises_error(sample_gdf):
    gdf = sample_gdf.drop(columns=["season"])
    with pytest.raises(ValueError, match="must have a 'season' column"):
        calculate_seasonal_home_range(gdf)


def test_default_groupby_and_percentiles_runs(sample_seasons_gdf):
    """Test that function runs with default parameters."""
    result = calculate_seasonal_home_range(sample_seasons_gdf)
    # Should return a DataFrame or GeoDataFrame
    assert isinstance(result, pd.DataFrame)
    # Should contain the default groupby columns in the result
    for col in ["groupby_col", "season"]:
        assert col in result.columns


def test_custom_groupby_cols(sample_seasons_gdf):
    custom_cols = ["groupby_cols", "season"]
    result = calculate_seasonal_home_range(sample_seasons_gdf, groupby_cols=custom_cols)
    # Result index/columns include custom groupby
    for col in custom_cols:
        assert col in result.columns


def test_auto_vs_custom_cell_size(sample_seasons_gdf):
    # Auto-scale (default)
    result_auto = calculate_seasonal_home_range(sample_seasons_gdf)
    assert isinstance(result_auto, pd.DataFrame)

    # Custom grid cell size
    custom_cell_size = CustomGridCellSize(cell_size=500)
    result_custom = calculate_seasonal_home_range(sample_seasons_gdf, auto_scale_or_custom_cell_size=custom_cell_size)
    assert isinstance(result_custom, pd.DataFrame)


def test_create_seasonal_labels_happy_path(sample_gdf, sample_seasons_df):
    """
    Test that trajectories are correctly labeled with seasonal windows.
    """
    result = create_seasonal_labels(sample_gdf, sample_seasons_df)

    assert result is not None
    assert "season" in result.columns
    # Check that all rows have a non-null season
    assert result["season"].notna().all()
    # Check that only valid seasons from the seasonal_df are assigned
    valid_seasons = sample_seasons_df["season"].unique()
    assert set(result["season"].unique()).issubset(set(valid_seasons))


def test_create_seasonal_labels_empty_trajectories(sample_seasons_df):
    """
    Test that empty trajectories GeoDataFrame raises an error.
    """
    empty_gdf = gpd.GeoDataFrame(columns=["segment_start", "segment_end", "geometry"])
    with pytest.raises(ValueError, match="trajectories gdf is empty"):
        create_seasonal_labels(empty_gdf, sample_seasons_df)
