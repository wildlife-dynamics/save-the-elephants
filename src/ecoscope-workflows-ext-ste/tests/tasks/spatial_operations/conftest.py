"""Shared fixtures for spatial_operations tests.

Loads the real geospatial fixture files under ``tests/data/`` once per
session and hands out fresh copies to each test (GeoDataFrames are mutated
in place by some of the functions under test, e.g. ``.copy()`` calls
notwithstanding, so we always return a ``.copy()`` from the fixture to keep
tests isolated from one another).
"""

from pathlib import Path

import geopandas as gpd
import pytest

DATA_DIR = Path(__file__).parents[3] / "tests" / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def _aois_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "AOIs.gpkg", layer="AOIs")


@pytest.fixture(scope="session")
def _kenya_pa_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "kenya_pa.gpkg")


@pytest.fixture(scope="session")
def _kenyan_counties_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "kenyan_counties.gpkg")


@pytest.fixture(scope="session")
def _sample_trajs_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "sample_trajs.gpkg")


@pytest.fixture(scope="session")
def _sample_season_traj_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "sample_season_traj.gpkg")


@pytest.fixture
def aois_gdf(_aois_raw) -> gpd.GeoDataFrame:
    """7 named AOI polygons, EPSG:4326. Columns: name, geometry."""
    return _aois_raw.copy()


@pytest.fixture
def kenya_pa_gdf(_kenya_pa_raw) -> gpd.GeoDataFrame:
    """117 Kenyan protected-area polygons, EPSG:4326.

    Columns: objectid, type, name, geometry.
    """
    return _kenya_pa_raw.copy()


@pytest.fixture
def kenyan_counties_gdf(_kenyan_counties_raw) -> gpd.GeoDataFrame:
    """47 Kenyan county polygons, EPSG:4326."""
    return _kenyan_counties_raw.copy()


@pytest.fixture
def sample_trajs_gdf(_sample_trajs_raw) -> gpd.GeoDataFrame:
    """1520 trajectory segment LineStrings near Samburu, EPSG:4326."""
    return _sample_trajs_raw.copy()


@pytest.fixture
def sample_season_traj_gdf(_sample_season_traj_raw) -> gpd.GeoDataFrame:
    """1520 trajectory segment LineStrings with a `season` column, EPSG:4326."""
    return _sample_season_traj_raw.copy()
