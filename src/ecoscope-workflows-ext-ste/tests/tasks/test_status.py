import pytest
import geopandas as gpd
from shapely.geometry import Point
from ecoscope_workflows_ext_ste.tasks._status import modify_status_colors, assign_status_colors


@pytest.fixture
def sample_current_gdf():
    """Fixture for current period GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "subject_name": ["Elephant1", "Elephant2"],
            "hex_color": ["#ff0000", "#00ff00"],
            "duration_status": ["Current tracks", "Current tracks"],
            "geometry": [Point(35.0, 1.0), Point(36.0, 1.0)],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_previous_gdf():
    """Fixture for previous period GeoDataFrame."""
    return gpd.GeoDataFrame(
        {
            "subject_name": ["Elephant1", "Elephant2"],
            "hex_color": ["#ff0000", "#00ff00"],
            "duration_status": ["Previous tracks", "Previous tracks"],
            "geometry": [Point(35.5, 1.5), Point(36.5, 1.5)],
        },
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_mixed_gdf():
    """Fixture for mixed current and previous periods."""
    return gpd.GeoDataFrame(
        {
            "subject_name": ["Elephant1", "Elephant1", "Elephant2", "Elephant2"],
            "hex_color": ["#ff0000", "#ff0000", "#00ff00", "#00ff00"],
            "subject_sex": ["Male", "Male", "Female", "Female"],
            "duration_status": ["Current tracks", "Previous tracks", "Current tracks", "Previous tracks"],
            "geometry": [Point(35.0, 1.0), Point(35.5, 1.5), Point(36.0, 1.0), Point(36.5, 1.5)],
        },
        crs="EPSG:4326",
    )


# Tests for assign_status_colors
def test_assign_current_uses_hex_column(sample_current_gdf):
    """Test that Current tracks use hex_column when use_hex_column_for_current=True."""
    result = assign_status_colors(
        gdf=sample_current_gdf, hex_column="hex_color", previous_color_hex="#808080", use_hex_column_for_current=True
    )

    assert "duration_status_hex_colors" in result.columns
    assert "duration_status_colors" in result.columns
    assert result.loc[0, "duration_status_hex_colors"] == "#ff0000"
    assert result.loc[1, "duration_status_hex_colors"] == "#00ff00"


def test_assign_previous_uses_gray(sample_previous_gdf):
    """Test that Previous tracks use gray color."""
    result = assign_status_colors(
        gdf=sample_previous_gdf, hex_column="hex_color", previous_color_hex="#808080", use_hex_column_for_current=True
    )

    assert all(result["duration_status_hex_colors"] == "#808080")


def test_assign_current_uses_default_hex(sample_current_gdf):
    """Test that Current tracks use default_current_hex when specified."""
    result = assign_status_colors(
        gdf=sample_current_gdf,
        hex_column="hex_color",
        previous_color_hex="#808080",
        use_hex_column_for_current=False,
        default_current_hex="#00008b",
    )

    assert all(result["duration_status_hex_colors"] == "#00008b")


def test_assign_mixed_current_and_previous(sample_mixed_gdf):
    """Test mixed current and previous tracks get correct colors."""
    result = assign_status_colors(
        gdf=sample_mixed_gdf, hex_column="hex_color", previous_color_hex="#808080", use_hex_column_for_current=True
    )

    # Current tracks should use hex_color
    assert result.loc[0, "duration_status_hex_colors"] == "#ff0000"
    assert result.loc[2, "duration_status_hex_colors"] == "#00ff00"

    # Previous tracks should be gray
    assert result.loc[1, "duration_status_hex_colors"] == "#808080"
    assert result.loc[3, "duration_status_hex_colors"] == "#808080"


def test_rgba_colors_created(sample_current_gdf):
    """Test that RGBA colors are properly created from hex."""
    result = assign_status_colors(
        gdf=sample_current_gdf, hex_column="hex_color", previous_color_hex="#808080", use_hex_column_for_current=True
    )

    # Check RGBA column exists and contains tuples/lists
    assert "duration_status_colors" in result.columns
    rgba = result.loc[0, "duration_status_colors"]
    assert isinstance(rgba, (tuple, list))
    assert len(rgba) == 4  # RGBA should have 4 values


def test_empty_gdf_raises_error():
    """Test that empty GeoDataFrame raises ValueError."""
    empty_gdf = gpd.GeoDataFrame()

    with pytest.raises(ValueError, match="gdf is empty"):
        assign_status_colors(gdf=empty_gdf, hex_column="hex_color", previous_color_hex="#808080")


def test_none_gdf_raises_error():
    """Test that None GeoDataFrame raises ValueError."""
    with pytest.raises(ValueError, match="gdf is empty"):
        assign_status_colors(gdf=None, hex_column="hex_color", previous_color_hex="#808080")


def test_missing_hex_column_raises_error(sample_current_gdf):
    """Test that missing hex_column raises ValueError."""
    with pytest.raises(ValueError, match="Missing column 'nonexistent'"):
        assign_status_colors(gdf=sample_current_gdf, hex_column="nonexistent", previous_color_hex="#808080")


def test_missing_duration_status_column_raises_error():
    """Test that missing duration_status column raises ValueError."""
    gdf = gpd.GeoDataFrame({"hex_color": ["#ff0000"], "geometry": [Point(35.0, 1.0)]})

    with pytest.raises(ValueError, match="Missing 'duration_status' column"):
        assign_status_colors(gdf=gdf, hex_column="hex_color", previous_color_hex="#808080")


def test_invalid_previous_color_hex_raises_error(sample_current_gdf):
    """Test that invalid previous_color_hex raises ValueError."""
    with pytest.raises(ValueError, match="Invalid previous_color_hex"):
        assign_status_colors(
            gdf=sample_current_gdf,
            hex_column="hex_color",
            previous_color_hex="red",  # Not a hex color
        )


def test_invalid_default_current_hex_raises_error(sample_current_gdf):
    """Test that invalid default_current_hex raises ValueError."""
    with pytest.raises(ValueError, match="Invalid default_current_hex"):
        assign_status_colors(
            gdf=sample_current_gdf,
            hex_column="hex_color",
            previous_color_hex="#808080",
            use_hex_column_for_current=False,
            default_current_hex="blue",  # Not a hex color
        )


# Tests for modify_status_colors


def test_subject_name_uses_individual_colors(sample_mixed_gdf):
    """Test that grouper_value='subject_name' uses individual subject colors."""
    result = modify_status_colors(grouper_value="subject_name", gdf=sample_mixed_gdf)

    # Current tracks should use hex_color column
    assert result.loc[0, "duration_status_hex_colors"] == "#ff0000"
    assert result.loc[2, "duration_status_hex_colors"] == "#00ff00"

    # Previous tracks should be gray
    assert result.loc[1, "duration_status_hex_colors"] == "#808080"
    assert result.loc[3, "duration_status_hex_colors"] == "#808080"


def test_subject_sex_uses_uniform_color(sample_mixed_gdf):
    """Test that grouper_value='subject_sex' uses uniform dark blue for current."""
    result = modify_status_colors(grouper_value="subject_sex", gdf=sample_mixed_gdf)

    # All current tracks should be dark blue
    assert result.loc[0, "duration_status_hex_colors"] == "#00008b"
    assert result.loc[2, "duration_status_hex_colors"] == "#00008b"

    # All previous tracks should be gray
    assert result.loc[1, "duration_status_hex_colors"] == "#808080"
    assert result.loc[3, "duration_status_hex_colors"] == "#808080"


def test_subject_subtype_uses_uniform_color(sample_mixed_gdf):
    """Test that grouper_value='subject_subtype' uses uniform color."""
    result = modify_status_colors(grouper_value="subject_subtype", gdf=sample_mixed_gdf)

    # Current tracks should be dark blue (uniform)
    current_mask = result["duration_status"] == "Current tracks"
    assert all(result.loc[current_mask, "duration_status_hex_colors"] == "#00008b")

    # Previous tracks should be gray
    previous_mask = result["duration_status"] == "Previous tracks"
    assert all(result.loc[previous_mask, "duration_status_hex_colors"] == "#808080")


def test_empty_grouper_value_raises_error(sample_mixed_gdf):
    """Test that empty grouper_value raises ValueError."""
    with pytest.raises(ValueError, match="grouper_value is empty"):
        modify_status_colors(grouper_value="", gdf=sample_mixed_gdf)


def test_whitespace_grouper_value_raises_error(sample_mixed_gdf):
    """Test that whitespace-only grouper_value raises ValueError."""
    with pytest.raises(ValueError, match="grouper_value is empty"):
        modify_status_colors(grouper_value="   ", gdf=sample_mixed_gdf)


def test_original_gdf_unchanged(sample_mixed_gdf):
    """Test that original GeoDataFrame is not modified."""

    result = modify_status_colors(grouper_value="subject_name", gdf=sample_mixed_gdf)

    # Original should not have new columns
    assert "duration_status_hex_colors" not in sample_mixed_gdf.columns
    assert "duration_status_colors" not in sample_mixed_gdf.columns

    # Result should have new columns
    assert "duration_status_hex_colors" in result.columns
    assert "duration_status_colors" in result.columns


# Integration Tests
def test_full_workflow_subject_name(sample_mixed_gdf):
    """Test complete workflow with subject_name grouping."""
    result = modify_status_colors(grouper_value="subject_name", gdf=sample_mixed_gdf)

    # Verify all required columns exist
    assert "duration_status_hex_colors" in result.columns
    assert "duration_status_colors" in result.columns

    # Verify color assignments
    assert len(result) == 4
    assert result["duration_status_hex_colors"].notna().all()
    assert result["duration_status_colors"].notna().all()


def test_full_workflow_subject_sex(sample_mixed_gdf):
    """Test complete workflow with subject_sex grouping."""
    result = modify_status_colors(grouper_value="subject_sex", gdf=sample_mixed_gdf)

    # All current should be uniform color
    current_colors = result[result["duration_status"] == "Current tracks"]["duration_status_hex_colors"]
    assert len(current_colors.unique()) == 1
    assert current_colors.iloc[0] == "#00008b"

    # All previous should be gray
    previous_colors = result[result["duration_status"] == "Previous tracks"]["duration_status_hex_colors"]
    assert len(previous_colors.unique()) == 1
    assert previous_colors.iloc[0] == "#808080"
