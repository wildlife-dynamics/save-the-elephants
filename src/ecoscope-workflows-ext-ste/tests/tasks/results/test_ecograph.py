import os
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from ecoscope_workflows_ext_ste.tasks.results._ecograph import generate_ecograph_raster

TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def sample_trajs_gdf():
    """Sample trajectory segments (LineStrings) with the columns expected by Ecograph/Trajectory."""
    return gpd.read_file(TEST_DATA_DIR / "sample_trajs.gpkg")


# ========================================================================
# Happy path
# ========================================================================


def test_generate_raster_with_movement_covariate(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_speed",
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)
    assert result.endswith(".tif")
    assert "test_speed" in result
    assert os.path.getsize(result) > 0


def test_generate_raster_with_network_metric(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_network",
        network_metric="weight",
        step_length=200,
    )

    assert os.path.exists(result)
    assert result.endswith(".tif")
    assert os.path.getsize(result) > 0


# ========================================================================
# Filename handling
# ========================================================================


def test_generate_raster_auto_filename_is_12_char_hash(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename=None,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)
    filename = os.path.basename(result).replace(".tif", "")
    assert len(filename) == 12


def test_generate_raster_auto_filename_is_deterministic(sample_trajs_gdf, tmp_path):
    """The same input data should always hash to the same filename."""
    result1 = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path / "a"),
        filename=None,
        movement_covariate="speed",
        step_length=200,
    )
    result2 = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path / "b"),
        filename=None,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.basename(result1) == os.path.basename(result2)


def test_generate_raster_custom_filename(sample_trajs_gdf, tmp_path):
    custom_name = "my_custom_raster"
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename=custom_name,
        movement_covariate="speed",
        step_length=200,
    )

    assert custom_name in result
    assert os.path.exists(result)


# ========================================================================
# Output directory handling
# ========================================================================


def test_generate_raster_default_output_dir_is_cwd(sample_trajs_gdf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=None,
        filename="test_default_dir",
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.dirname(result) == str(tmp_path)
    assert os.path.exists(result)


def test_generate_raster_creates_nested_output_dir(sample_trajs_gdf, tmp_path):
    nested_dir = tmp_path / "nested" / "dir"

    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(nested_dir),
        filename="test_nested",
        movement_covariate="speed",
        step_length=200,
    )

    assert nested_dir.exists()
    assert os.path.exists(result)


# ========================================================================
# Resolution / step_length precedence
# ========================================================================


def test_generate_raster_with_custom_resolution(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_resolution",
        resolution=200.0,
        movement_covariate="speed",
    )

    assert os.path.exists(result)


def test_generate_raster_with_step_length(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_step",
        step_length=150,
        movement_covariate="speed",
    )

    assert os.path.exists(result)


def test_generate_raster_auto_resolution_from_dist_col(sample_trajs_gdf, tmp_path):
    """When both resolution and step_length are None, resolution falls back to
    the mean of dist_col."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_auto_res",
        movement_covariate="speed",
        resolution=None,
        step_length=None,
    )

    assert os.path.exists(result)


# ========================================================================
# Interpolation / covariate / metric coverage
# ========================================================================


@pytest.mark.parametrize("interpolation", ["mean", "min", "max", "median"])
def test_generate_raster_interpolation_methods(sample_trajs_gdf, tmp_path, interpolation):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename=f"test_{interpolation}",
        interpolation=interpolation,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)


@pytest.mark.parametrize(
    "covariate",
    ["dot_product", "step_length", "speed", "sin_time", "cos_time", "tortuosity_1", "tortuosity_2"],
)
def test_generate_raster_movement_covariates(sample_trajs_gdf, tmp_path, covariate):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename=f"test_{covariate}",
        movement_covariate=covariate,
        step_length=200,
    )

    assert os.path.exists(result)


@pytest.mark.parametrize("metric", ["weight", "betweenness", "degree", "collective_influence"])
def test_generate_raster_network_metrics(sample_trajs_gdf, tmp_path, metric):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename=f"test_{metric}",
        network_metric=metric,
        step_length=200,
    )

    assert os.path.exists(result)


# ========================================================================
# Ecograph tuning parameters
# ========================================================================


def test_generate_raster_with_custom_radius(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_radius",
        radius=3,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)


def test_generate_raster_with_cutoff(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_cutoff",
        cutoff=500.0,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)


def test_generate_raster_with_tortuosity_length(sample_trajs_gdf, tmp_path):
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        output_dir=str(tmp_path),
        filename="test_tortuosity",
        tortuosity_length=5,
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)


# ========================================================================
# Parameter validation errors
# ========================================================================


def test_both_covariate_and_metric_raises_error(sample_trajs_gdf, tmp_path):
    with pytest.raises(ValueError, match="Provide exactly one"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            output_dir=str(tmp_path),
            movement_covariate="speed",
            network_metric="weight",
            step_length=200,
        )


def test_neither_covariate_nor_metric_raises_error(sample_trajs_gdf, tmp_path):
    with pytest.raises(ValueError, match="Provide exactly one"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            output_dir=str(tmp_path),
            step_length=200,
        )


def test_zero_resolution_raises_error(sample_trajs_gdf, tmp_path):
    with pytest.raises(ValueError, match="must be > 0"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            output_dir=str(tmp_path),
            resolution=0,
            movement_covariate="speed",
        )


def test_negative_resolution_raises_error(sample_trajs_gdf, tmp_path):
    with pytest.raises(ValueError, match="must be > 0"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            output_dir=str(tmp_path),
            resolution=-100,
            movement_covariate="speed",
        )


def test_missing_dist_col_raises_error(sample_trajs_gdf, tmp_path):
    with pytest.raises(KeyError):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            dist_col="nonexistent",
            output_dir=str(tmp_path),
            movement_covariate="speed",
        )


def test_non_numeric_dist_col_raises_error(tmp_path):
    gdf = gpd.GeoDataFrame(
        {
            "dist_meters": ["a", "b", "c"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="has no numeric values"):
        generate_ecograph_raster(gdf=gdf, output_dir=str(tmp_path), movement_covariate="speed")


def test_empty_gdf_raises_error(tmp_path):
    """An empty GeoDataFrame has no dist_meters column, so this fails with a
    KeyError while computing resolution (there is no dedicated "empty" check
    in the current implementation)."""
    empty_gdf = gpd.GeoDataFrame()

    with pytest.raises(KeyError):
        generate_ecograph_raster(gdf=empty_gdf, output_dir=str(tmp_path), movement_covariate="speed")


def test_none_gdf_raises_error(tmp_path):
    """Passing gdf=None fails with a TypeError (no subscript support), since
    there is no explicit None-check in the current implementation."""
    with pytest.raises(TypeError):
        generate_ecograph_raster(gdf=None, output_dir=str(tmp_path), movement_covariate="speed")


# ========================================================================
# Edge cases
# ========================================================================


def test_single_row_gdf(sample_trajs_gdf, tmp_path):
    single = sample_trajs_gdf.iloc[[0]].reset_index(drop=True)
    result = generate_ecograph_raster(
        gdf=single,
        output_dir=str(tmp_path),
        movement_covariate="speed",
        step_length=200,
    )

    assert os.path.exists(result)


def test_missing_crs_raises_error(sample_trajs_gdf, tmp_path):
    """Ecograph/Trajectory need a CRS to estimate the UTM projection used for
    distance calculations; a gdf without a CRS fails downstream."""
    gdf_no_crs = sample_trajs_gdf.copy()
    gdf_no_crs.crs = None

    with pytest.raises(Exception, match="crs must be set"):
        generate_ecograph_raster(
            gdf=gdf_no_crs,
            output_dir=str(tmp_path),
            movement_covariate="speed",
            step_length=200,
        )
