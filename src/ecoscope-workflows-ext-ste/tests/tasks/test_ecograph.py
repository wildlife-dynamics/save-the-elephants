import pytest
import os
import geopandas as gpd
from pathlib import Path
from tempfile import TemporaryDirectory
from shapely.geometry import Point
from unittest.mock import patch

from ecoscope_workflows_ext_ste.tasks._ecograph import generate_ecograph_raster, retrieve_feature_gdf

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def sample_trajs_gdf():
    """Fixture to load the sample trajectories GeoDataFrame."""
    return gpd.read_file(TEST_DATA_DIR, "sample_trajs.gpkg").to_crs("EPSG:4326")


@pytest.fixture
def temp_dir():
    """Fixture to create a temporary directory."""
    with TemporaryDirectory() as tmp:
        yield tmp


# Tests for generate_ecograph_raster


def test_generate_raster_with_movement_covariate(sample_trajs_gdf, temp_dir):
    """Test generating raster with movement covariate (speed)."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_speed",
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)
    assert result.endswith(".tif")
    assert "test_speed" in result
    assert os.path.getsize(result) > 0


@pytest.mark.integration
def test_generate_raster_with_real_data(sample_trajs_gdf, temp_dir):
    """Test generating raster with real trajectory data."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="real_data_test",
        movement_covariate="speed",
        step_length=2000,
        radius=2,
    )

    assert os.path.exists(result)
    assert result.endswith(".tif")
    assert os.path.getsize(result) > 0


def test_generate_raster_with_network_metric(sample_trajs_gdf, temp_dir):
    """Test generating raster with network metric."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_network",
        network_metric="weight",
        step_length=100,
    )

    assert os.path.exists(result)
    assert result.endswith(".tif")


# ========================================================================
# Filename handling tests
# ========================================================================
def test_generate_raster_auto_filename(sample_trajs_gdf, temp_dir):
    """Test that auto-generated filename is created when not provided."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename=None,  # Let it auto-generate
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)
    # Filename should be a 7-character hash
    filename = os.path.basename(result).replace(".tif", "")
    assert len(filename) == 7


def test_generate_raster_custom_filename(sample_trajs_gdf, temp_dir):
    """Test custom filename is used."""
    custom_name = "my_custom_raster"
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename=custom_name,
        movement_covariate="speed",
        step_length=100,
    )

    assert custom_name in result
    assert os.path.exists(result)


# Output directory tests
def test_generate_raster_default_output_dir(sample_trajs_gdf):
    """Test that default output directory is current working directory."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=None,  # Should default to cwd
        filename="test_default_dir",
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.dirname(result) == os.getcwd()
    # Cleanup
    if os.path.exists(result):
        os.remove(result)


def test_generate_raster_creates_output_dir(sample_trajs_gdf, temp_dir):
    """Test that output directory is created if it doesn't exist."""
    nested_dir = os.path.join(temp_dir, "nested", "dir")

    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=nested_dir,
        filename="test_nested",
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(nested_dir)
    assert os.path.exists(result)


def test_generate_raster_removes_file_scheme(sample_trajs_gdf, temp_dir):
    """Test that file:// scheme is removed from output_dir."""
    file_scheme_path = f"file://{temp_dir}"

    with patch("your_module.remove_file_scheme") as mock_remove:
        mock_remove.return_value = temp_dir

        result = generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            dist_col="dist_meters",
            output_dir=file_scheme_path,
            filename="test_scheme",
            movement_covariate="speed",
            step_length=100,
        )
        assert os.path.exists(result)

        mock_remove.assert_called_once_with(file_scheme_path)


# Resolution and step_length tests
def test_generate_raster_with_custom_resolution(sample_trajs_gdf, temp_dir):
    """Test using custom resolution."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_resolution",
        resolution=200.0,
        movement_covariate="speed",
    )

    assert os.path.exists(result)


def test_generate_raster_with_step_length(sample_trajs_gdf, temp_dir):
    """Test using step_length parameter."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_step",
        step_length=150,
        movement_covariate="speed",
    )

    assert os.path.exists(result)


def test_generate_raster_auto_resolution_from_dist_col(sample_trajs_gdf, temp_dir):
    """Test that resolution is computed from dist_col when not provided."""
    # Don't provide resolution or step_length - should use mean of dist_meters
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_auto_res",
        movement_covariate="speed",
    )

    assert os.path.exists(result)
    # Resolution should be mean of [100, 150, 120, 180] = 137.5


# Interpolation tests
@pytest.mark.parametrize("interpolation", ["mean", "min", "max", "median"])
def test_generate_raster_interpolation_methods(sample_trajs_gdf, temp_dir, interpolation):
    """Test different interpolation methods."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename=f"test_{interpolation}",
        interpolation=interpolation,
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)


# Movement covariate tests
@pytest.mark.parametrize(
    "covariate", ["dot_product", "step_length", "speed", "sin_time", "cos_time", "tortuosity_1", "tortuosity_2"]
)
def test_generate_raster_movement_covariates(sample_trajs_gdf, temp_dir, covariate):
    """Test different movement covariates."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename=f"test_{covariate}",
        movement_covariate=covariate,
        step_length=100,
    )

    assert os.path.exists(result)


# Network metric tests
@pytest.mark.parametrize("metric", ["weight", "betweenness", "degree", "collective_influence"])
def test_generate_raster_network_metrics(sample_trajs_gdf, temp_dir, metric):
    """Test different network metrics."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename=f"test_{metric}",
        network_metric=metric,
        step_length=100,
    )

    assert os.path.exists(result)


# Parameter validation tests
def test_generate_raster_both_covariate_and_metric_raises_error(sample_trajs_gdf, temp_dir):
    """Test that providing both movement_covariate and network_metric raises error."""
    with pytest.raises(ValueError, match="Provide exactly one"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            dist_col="dist_meters",
            output_dir=temp_dir,
            filename="test_both",
            movement_covariate="speed",
            network_metric="weight",
            step_length=100,
        )


def test_generate_raster_neither_covariate_nor_metric_raises_error(sample_trajs_gdf, temp_dir):
    """Test that providing neither movement_covariate nor network_metric raises error."""
    with pytest.raises(ValueError, match="Provide exactly one"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf, dist_col="dist_meters", output_dir=temp_dir, filename="test_neither", step_length=100
        )


# Error handling tests
def test_generate_raster_empty_gdf_raises_error(temp_dir):
    """Test that empty GeoDataFrame raises error."""
    empty_gdf = gpd.GeoDataFrame()

    with pytest.raises(ValueError, match="Trajectory gdf is empty"):
        generate_ecograph_raster(gdf=empty_gdf, dist_col="dist_meters", output_dir=temp_dir, movement_covariate="speed")


def test_generate_raster_none_gdf_raises_error(temp_dir):
    """Test that None GeoDataFrame raises error."""
    with pytest.raises(ValueError, match="Trajectory gdf is empty"):
        generate_ecograph_raster(gdf=None, dist_col="dist_meters", output_dir=temp_dir, movement_covariate="speed")


def test_generate_raster_missing_dist_col_raises_error(sample_trajs_gdf, temp_dir):
    """Test that missing dist_col raises error."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf, dist_col="nonexistent", output_dir=temp_dir, movement_covariate="speed"
        )


def test_generate_raster_non_numeric_dist_col_raises_error(temp_dir):
    """Test that non-numeric dist_col raises error."""
    gdf = gpd.GeoDataFrame(
        {
            "dist_meters": ["a", "b", "c"],  # Non-numeric
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="has no numeric values"):
        generate_ecograph_raster(gdf=gdf, dist_col="dist_meters", output_dir=temp_dir, movement_covariate="speed")


def test_generate_raster_zero_resolution_raises_error(sample_trajs_gdf, temp_dir):
    """Test that zero resolution raises error."""
    with pytest.raises(ValueError, match="must be > 0"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf, dist_col="dist_meters", output_dir=temp_dir, resolution=0, movement_covariate="speed"
        )


def test_generate_raster_negative_resolution_raises_error(sample_trajs_gdf, temp_dir):
    """Test that negative resolution raises error."""
    with pytest.raises(ValueError, match="must be > 0"):
        generate_ecograph_raster(
            gdf=sample_trajs_gdf,
            dist_col="dist_meters",
            output_dir=temp_dir,
            resolution=-100,
            movement_covariate="speed",
        )


# Ecograph parameter tests
def test_generate_raster_with_custom_radius(sample_trajs_gdf, temp_dir):
    """Test using custom radius parameter."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_radius",
        radius=3,
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)


def test_generate_raster_with_cutoff(sample_trajs_gdf, temp_dir):
    """Test using cutoff parameter."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_cutoff",
        cutoff=500.0,
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)


def test_generate_raster_with_tortuosity_length(sample_trajs_gdf, temp_dir):
    """Test using tortuosity_length parameter."""
    result = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_tortuosity",
        tortuosity_length=5,
        movement_covariate="speed",
        step_length=100,
    )

    assert os.path.exists(result)


# Tests for retrieve_feature_gdf
# ============================================================================


def test_retrieve_feature_from_generated_raster(sample_trajs_gdf, temp_dir):
    """Test retrieving features from a generated raster."""
    # First generate a raster
    raster_path = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_retrieve",
        movement_covariate="speed",
        step_length=100,
    )

    # Then retrieve features
    result_gdf = retrieve_feature_gdf(file_path=raster_path)

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert len(result_gdf) > 0
    assert result_gdf.crs is not None


@pytest.mark.integration
def test_retrieve_feature_real_workflow(sample_trajs_gdf, temp_dir):
    """Test full workflow: generate raster then retrieve features."""
    # Generate raster
    raster_path = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="real_workflow",
        movement_covariate="speed",
        step_length=2000,
    )

    # Retrieve features
    result_gdf = retrieve_feature_gdf(file_path=raster_path)

    assert isinstance(result_gdf, gpd.GeoDataFrame)
    assert len(result_gdf) > 0
    assert "value" in result_gdf.columns
    assert result_gdf.geometry.notnull().all()


def test_retrieve_feature_empty_string_raises_error():
    """Test that empty string file_path raises error."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        retrieve_feature_gdf(file_path="")


def test_retrieve_feature_none_raises_error():
    """Test that None file_path raises error."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        retrieve_feature_gdf(file_path=None)


def test_retrieve_feature_non_string_raises_error():
    """Test that non-string file_path raises error."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        retrieve_feature_gdf(file_path=123)


def test_retrieve_feature_has_value_column(sample_trajs_gdf, temp_dir):
    """Test that retrieved features have 'value' column."""
    raster_path = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_value_col",
        movement_covariate="speed",
        step_length=100,
    )

    result_gdf = retrieve_feature_gdf(file_path=raster_path)

    assert "value" in result_gdf.columns


def test_retrieve_feature_has_geometry(sample_trajs_gdf, temp_dir):
    """Test that retrieved features have valid geometries."""
    raster_path = generate_ecograph_raster(
        gdf=sample_trajs_gdf,
        dist_col="dist_meters",
        output_dir=temp_dir,
        filename="test_geom",
        movement_covariate="speed",
        step_length=100,
    )

    result_gdf = retrieve_feature_gdf(file_path=raster_path)

    assert result_gdf.geometry.notnull().all()
    assert all(result_gdf.geometry.is_valid)
