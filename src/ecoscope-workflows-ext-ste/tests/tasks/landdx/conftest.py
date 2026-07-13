"""Shared fixtures for landdx tests.

Loads the real `kenya_pa.gpkg` fixture (columns: objectid, type, name,
geometry -- the same `type`/`name`/`geometry` shape LandDx layers are
expected to have) once per session and hands out fresh copies / the raw
file path to each test.
"""

from pathlib import Path

import geopandas as gpd
import pytest

DATA_DIR = Path(__file__).parents[3] / "tests" / "data"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def kenya_pa_path() -> Path:
    """Path to a real GeoPackage with `type`/`name`/`geometry` columns.

    `type` values are 'Community Conservancy', 'National Park', and
    'National Reserve' -- exactly the three keys LandDx layers colour by.
    """
    return DATA_DIR / "kenya_pa.gpkg"


@pytest.fixture(scope="session")
def _kenya_pa_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "kenya_pa.gpkg")


@pytest.fixture
def kenya_pa_gdf(_kenya_pa_raw) -> gpd.GeoDataFrame:
    """117 Kenyan protected-area polygons, EPSG:4326.

    Columns: objectid, type, name, geometry.
    """
    return _kenya_pa_raw.copy()
