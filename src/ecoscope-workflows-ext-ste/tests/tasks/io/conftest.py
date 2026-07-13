"""Shared fixtures for `ecoscope_workflows_ext_ste.tasks.io` tests.

Loads the real geospatial fixture files under ``tests/data/`` once per
session and hands out a fresh copy to each test, mirroring the pattern used
by the sibling `tests/tasks/spatial_operations` and `tests/tasks/transformation`
suites.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest

DATA_DIR = Path(__file__).parents[3] / "tests" / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def _aois_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "AOIs.gpkg", layer="AOIs")


@pytest.fixture(scope="session")
def _sample_trajs_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "sample_trajs.gpkg")


@pytest.fixture(scope="session")
def _sample_season_traj_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "sample_season_traj.gpkg")


@pytest.fixture(scope="session")
def _seasonal_windows_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "seasonal_windows.csv")
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


@pytest.fixture
def aois_gdf(_aois_raw) -> gpd.GeoDataFrame:
    """7 named AOI polygons, EPSG:4326. Columns: name, geometry."""
    return _aois_raw.copy()


@pytest.fixture
def sample_trajs_gdf(_sample_trajs_raw) -> gpd.GeoDataFrame:
    """1520 trajectory segment LineStrings near Samburu, EPSG:4326.

    Columns: id, groupby_col, segment_start, segment_end, timespan_seconds,
    dist_meters, speed_kmhr, heading, junk_status, nsd, geometry.
    """
    return _sample_trajs_raw.copy()


@pytest.fixture
def sample_season_traj_gdf(_sample_season_traj_raw) -> gpd.GeoDataFrame:
    """1520 trajectory segment LineStrings with a `season` column, EPSG:4326."""
    return _sample_season_traj_raw.copy()


@pytest.fixture
def seasonal_windows_df(_seasonal_windows_raw) -> pd.DataFrame:
    """Seasonal windows with `start`, `end`, `season` (tz-aware UTC) columns."""
    return _seasonal_windows_raw.copy()
