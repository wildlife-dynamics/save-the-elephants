"""Tests for ecoscope_workflows_ext_ste.tasks.io._raster.

`raster_to_gdf` is registered via `wt_registry.register()`, which is a no-op
at call time, so it behaves as a plain Python function here.

Rather than hand-crafting a synthetic raster, the module-scoped
`ecograph_raster_path` fixture builds a *real* Ecograph feature GeoTIFF via
the sibling `results._ecograph.generate_ecograph_raster` task (using the
`sample_trajs.gpkg` fixture data), the same way the file would actually be
produced in a real workflow. `raster_to_gdf` wraps `get_feature_gdf` from
`ecoscope.analysis.ecograph`, which is exactly what
`generate_ecograph_raster` writes -- so this is the natural real-file
counterpart to `raster_to_gdf`, mirroring the old (pre-reorg) test suite's
"generate then retrieve" pattern.
"""

import geopandas as gpd
import pytest

from ecoscope_workflows_ext_ste.tasks.io._raster import raster_to_gdf
from ecoscope_workflows_ext_ste.tasks.results._ecograph import generate_ecograph_raster


@pytest.fixture(scope="module")
def ecograph_raster_path(tmp_path_factory, _sample_trajs_raw) -> str:
    """A small, real Ecograph feature GeoTIFF, generated once for this module."""
    out_dir = tmp_path_factory.mktemp("raster")
    # Use a small subset + coarse resolution so this stays fast.
    gdf = _sample_trajs_raw.iloc[:80].copy()
    return generate_ecograph_raster(
        gdf=gdf,
        output_dir=str(out_dir),
        filename="feature",
        movement_covariate="speed",
        step_length=500,
    )


class TestRasterToGdf:
    def test_loads_real_feature_raster(self, ecograph_raster_path):
        result = raster_to_gdf(file_path=ecograph_raster_path)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0
        assert "value" in result.columns
        assert "geometry" in result.columns
        assert result.geometry.notnull().all()

    def test_accepts_file_scheme_prefixed_path(self, ecograph_raster_path):
        uri = "file://" + ecograph_raster_path

        result = raster_to_gdf(file_path=uri)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            raster_to_gdf(file_path="")

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            raster_to_gdf(file_path="   ")

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            raster_to_gdf(file_path=None)

    def test_nonexistent_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "does_not_exist.tif"

        with pytest.raises(FileNotFoundError, match="Feature file not found"):
            raster_to_gdf(file_path=str(missing))
