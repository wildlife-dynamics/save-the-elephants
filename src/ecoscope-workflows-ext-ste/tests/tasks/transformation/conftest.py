"""Shared fixtures for transformation tests.

Loads the real geospatial fixture file under ``tests/data/`` once per
session and hands out a fresh copy to each test (functions under test call
``.copy()`` themselves in places, but we still isolate at the fixture level
so no test can leak mutations into another).
"""

from pathlib import Path

import geopandas as gpd
import pytest

DATA_DIR = Path(__file__).parents[3] / "tests" / "data"


@pytest.fixture(scope="session")
def _sample_trajs_raw() -> gpd.GeoDataFrame:
    return gpd.read_file(DATA_DIR / "sample_trajs.gpkg")


@pytest.fixture
def sample_trajs_gdf(_sample_trajs_raw) -> gpd.GeoDataFrame:
    """1520 trajectory segment LineStrings near Samburu, EPSG:4326.

    Columns: id, groupby_col, segment_start, segment_end, timespan_seconds,
    dist_meters, speed_kmhr, heading, junk_status, nsd, geometry.
    """
    return _sample_trajs_raw.copy()
