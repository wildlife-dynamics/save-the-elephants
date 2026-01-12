import pytest
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from shapely.geometry import Point
from ecoscope_workflows_core.tasks.filter._filter import TimeRange, UTC_TIMEZONEINFO, DEFAULT_TIME_FORMAT
from ecoscope_workflows_ext_ste.tasks import (
    split_gdf_by_column,
    generate_mcp_gdf,
    round_off_values,
    dataframe_column_first_unique_str,
    get_duration,
    filter_df_cols,
    create_column,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def kenyan_counties_gdf():
    path = TEST_DATA_DIR / "kenyan_counties.gpkg"

    print(f"Loading Kenyan counties from: {path}")
    print(f"File exists? {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(f"Missing test data file: {path}")

    gdf = gpd.read_file(path)

    print(f"Loaded {len(gdf)} rows")
    print(f"CRS: {gdf.crs}")
    print(f"Has valid geometries? {gdf.geometry.is_valid.all()}")
    print(f"Empty geometries count: {(gdf.geometry.is_empty).sum()}")

    if len(gdf) == 0:
        raise ValueError("kenyan_counties.gpkg is empty!")

    if gdf.geometry.is_empty.all() or gdf.geometry.isna().all():
        raise ValueError("All geometries are empty or NaN!")

    # This line will crash if the gdf is really broken
    print(f"Total bounds: {gdf.total_bounds}")

    return gdf


@pytest.fixture
def sample_point_gdf():
    """Fixture to create a simple point GeoDataFrame for MCP testing."""
    points = [
        Point(35.0, 1.0),
        Point(36.0, 1.0),
        Point(36.0, 2.0),
        Point(35.0, 2.0),
    ]
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "category": ["A", "A", "B", "B"]},
        geometry=points,
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_df():
    """Fixture for a simple pandas DataFrame."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["Nairobi", "Mombasa", "Kisumu"],
        }
    )


# create column tests
def test_create_new_column_int(sample_df):
    result = create_column(sample_df, "new_col", 42)

    assert "new_col" in result.columns
    assert all(result["new_col"] == 42)
    assert len(result.columns) == len(sample_df.columns) + 1


def test_create_new_column_float(sample_df):
    result = create_column(sample_df, "score", 98.5)
    assert "score" in result.columns
    assert all(result["score"] == 98.5)


def test_create_new_column_string(sample_df):
    result = create_column(sample_df, "status", "active")

    assert "status" in result.columns
    assert all(result["status"] == "active")


def test_column_already_exists(sample_df):
    original_values = sample_df["age"].copy()
    result = create_column(sample_df, "age", 999)

    assert all(result["age"] == original_values)
    assert not all(result["age"] == 999)


def test_create_column_geodataframe(kenyan_counties_gdf):
    """Test creating a column in a GeoDataFrame."""
    result = create_column(kenyan_counties_gdf, "region", "East Africa")

    assert "region" in result.columns
    assert all(result["region"] == "East Africa")
    assert isinstance(result, gpd.GeoDataFrame)


def test_original_df_unchanged(sample_df):
    """Test that original DataFrame is not modified."""
    original_cols = sample_df.columns.tolist()
    _ = create_column(sample_df, "new_col", 100)

    assert sample_df.columns.tolist() == original_cols
    assert "new_col" not in sample_df.columns


# Split gdf by column
def test_split_by_county(kenyan_counties_gdf):
    """Test splitting by COUNTY column."""
    result = split_gdf_by_column(kenyan_counties_gdf, "COUNTY")

    assert isinstance(result, dict)
    assert len(result) > 0
    assert "Turkana" in result
    assert "Marsabit" in result

    # Check that each split is a GeoDataFrame
    for county, gdf in result.items():
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert all(gdf["COUNTY"] == county)


def test_split_returns_correct_counts(kenyan_counties_gdf):
    """Test that split returns correct number of groups."""
    unique_counties = kenyan_counties_gdf["COUNTY"].nunique()
    result = split_gdf_by_column(kenyan_counties_gdf, "COUNTY")

    assert len(result) == unique_counties


def test_split_preserves_all_rows(kenyan_counties_gdf):
    """Test that all rows are preserved after split."""
    result = split_gdf_by_column(kenyan_counties_gdf, "COUNTY")

    total_rows = sum(len(gdf) for gdf in result.values())
    assert total_rows == len(kenyan_counties_gdf)


def test_split_simple_gdf(sample_point_gdf):
    """Test splitting a simple GeoDataFrame."""
    result = split_gdf_by_column(sample_point_gdf, "category")

    assert len(result) == 2
    assert "A" in result
    assert "B" in result
    assert len(result["A"]) == 2
    assert len(result["B"]) == 2


def test_split_by_nonexistent_column(kenyan_counties_gdf):
    """Test error when column doesn't exist."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        split_gdf_by_column(kenyan_counties_gdf, "nonexistent")


def test_split_empty_gdf():
    """Test error when GeoDataFrame is empty."""
    empty_gdf = gpd.GeoDataFrame()

    with pytest.raises(ValueError, match="gdf is empty"):
        split_gdf_by_column(empty_gdf, "COUNTY")


def test_split_none_gdf():
    """Test error when GeoDataFrame is None."""
    with pytest.raises(ValueError, match="gdf is empty"):
        split_gdf_by_column(None, "COUNTY")


# generate_mcp_gdf
def test_generate_mcp_from_points(sample_point_gdf):
    """Test MCP generation from point geometries."""
    result = generate_mcp_gdf(sample_point_gdf)

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert "area_m2" in result.columns
    assert "area_km2" in result.columns
    assert "mcp" in result.columns
    assert result["area_m2"].iloc[0] > 0
    assert result["area_km2"].iloc[0] > 0


def test_mcp_area_conversion(sample_point_gdf):
    """Test that area conversion from m2 to km2 is correct."""
    result = generate_mcp_gdf(sample_point_gdf)

    area_m2 = result["area_m2"].iloc[0]
    area_km2 = result["area_km2"].iloc[0]

    assert abs(area_km2 - area_m2 / 1_000_000) < 0.001


def test_mcp_crs_preserved(sample_point_gdf):
    """Test that original CRS is preserved."""
    original_crs = sample_point_gdf.crs
    result = generate_mcp_gdf(sample_point_gdf)

    assert result.crs == original_crs


def test_mcp_geometry_type(sample_point_gdf):
    """Test that output geometry is a Polygon."""
    result = generate_mcp_gdf(sample_point_gdf)

    assert result.geometry.iloc[0].geom_type == "Polygon"


def test_mcp_custom_planar_crs(sample_point_gdf):
    """Test MCP with custom planar CRS."""
    result = generate_mcp_gdf(sample_point_gdf, planar_crs="EPSG:32636")

    assert isinstance(result, gpd.GeoDataFrame)
    assert result["area_m2"].iloc[0] > 0


def test_mcp_empty_gdf():
    """Test error when GeoDataFrame is empty."""
    empty_gdf = gpd.GeoDataFrame()

    with pytest.raises(ValueError, match="gdf is empty"):
        generate_mcp_gdf(empty_gdf)


def test_mcp_none_gdf():
    """Test error when GeoDataFrame is None."""
    with pytest.raises(ValueError, match="gdf is empty"):
        generate_mcp_gdf(None)


def test_mcp_no_crs(sample_point_gdf):
    """Test error when CRS is not set."""
    gdf_no_crs = sample_point_gdf.copy()
    gdf_no_crs.crs = None

    with pytest.raises(ValueError, match="must have a CRS set"):
        generate_mcp_gdf(gdf_no_crs)


def test_mcp_with_invalid_geometries(sample_point_gdf):
    """Test MCP with some invalid geometries."""
    gdf_with_invalid = sample_point_gdf.copy()
    gdf_with_invalid.loc[0, "geometry"] = None

    # Should still work with valid geometries
    result = generate_mcp_gdf(gdf_with_invalid)
    assert isinstance(result, gpd.GeoDataFrame)
    assert result["area_m2"].iloc[0] > 0


# filter_df_cols
def test_filter_single_column(sample_df):
    """Test filtering a single column."""
    result = filter_df_cols(sample_df, "name")

    assert list(result.columns) == ["name"]
    assert len(result) == len(sample_df)


def test_filter_multiple_columns(sample_df):
    """Test filtering multiple columns."""
    result = filter_df_cols(sample_df, ["name", "city"])

    assert list(result.columns) == ["name", "city"]
    assert len(result) == len(sample_df)


def test_filter_all_columns(sample_df):
    """Test filtering with all columns."""
    all_cols = sample_df.columns.tolist()
    result = filter_df_cols(sample_df, all_cols)

    assert list(result.columns) == all_cols
    assert result.equals(sample_df)


def test_filter_geodataframe(kenyan_counties_gdf):
    """Test filtering columns from a GeoDataFrame."""
    result = filter_df_cols(kenyan_counties_gdf, ["COUNTY", "AREA", "geometry"])

    assert list(result.columns) == ["COUNTY", "AREA", "geometry"]
    assert isinstance(result, gpd.GeoDataFrame)


def test_filter_nonexistent_column(sample_df):
    """Test error when column doesn't exist."""
    with pytest.raises(ValueError, match="Columns not found"):
        filter_df_cols(sample_df, "nonexistent")


def test_filter_mixed_existent_nonexistent(sample_df):
    """Test error when some columns don't exist."""
    with pytest.raises(ValueError, match="Columns not found.*nonexistent.*missing"):
        filter_df_cols(sample_df, ["name", "nonexistent", "missing"])


def test_filter_preserves_order(sample_df):
    """Test that column order is preserved."""
    result = filter_df_cols(sample_df, ["city", "name"])

    assert list(result.columns) == ["city", "name"]


def test_error_message_shows_available_columns(sample_df):
    """Test that error message shows available columns."""
    try:
        filter_df_cols(sample_df, "nonexistent")
    except ValueError as e:
        assert "Available columns" in str(e)
        assert "name" in str(e)
        assert "age" in str(e)


# dataframe_column_first_unique_str
def test_first_unique_value(sample_df):
    """Test getting first unique value."""
    result = dataframe_column_first_unique_str(sample_df, "name")

    assert result == "Alice"
    assert isinstance(result, str)


def test_sentence_case_conversion():
    """Test that value is converted to sentence case."""
    df = pd.DataFrame({"status": ["ACTIVE", "inactive", "pending"]})
    result = dataframe_column_first_unique_str(df, "status")

    assert result == "Active"


def test_lowercase_conversion():
    """Test lowercase conversion."""
    df = pd.DataFrame({"county": ["turkana", "marsabit"]})
    result = dataframe_column_first_unique_str(df, "county")

    assert result == "Turkana"


def test_from_geodataframe(kenyan_counties_gdf):
    """Test with GeoDataFrame."""
    result = dataframe_column_first_unique_str(kenyan_counties_gdf, "COUNTY")

    assert isinstance(result, str)
    assert result in ["Turkana", "Marsabit", "Mandera", "Wajir", "West pokot"]


def test_single_value_column():
    """Test with column having single value."""
    df = pd.DataFrame({"region": ["East Africa", "East Africa", "East Africa"]})
    result = dataframe_column_first_unique_str(df, "region")

    assert result == "East africa"


def test_numeric_column():
    """Test with numeric column (should convert to string)."""
    df = pd.DataFrame({"year": [2024, 2023, 2022]})
    result = dataframe_column_first_unique_str(df, "year")

    assert result == "2024"
    assert isinstance(result, str)


def test_empty_dataframe():
    """Test error with empty DataFrame."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="df is empty"):
        dataframe_column_first_unique_str(empty_df, "any_column")


def test_none_dataframe():
    """Test error with None DataFrame."""
    with pytest.raises(ValueError, match="df is empty"):
        dataframe_column_first_unique_str(None, "any_column")


def test_nonexistent_column(sample_df):
    """Test error when column doesn't exist."""
    with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
        dataframe_column_first_unique_str(sample_df, "nonexistent")


def test_column_with_empty_values():
    """Test behavior with column containing empty strings."""
    df = pd.DataFrame({"status": ["", "active", "pending"]})
    result = dataframe_column_first_unique_str(df, "status")

    # First unique value is empty string
    assert result == ""


# round_off_values
def test_round_to_zero_decimals():
    """Test rounding to zero decimal places."""
    result = round_off_values(3.7, 0)
    assert result == 4.0
    assert isinstance(result, float)


def test_round_to_one_decimal():
    """Test rounding to one decimal place."""
    result = round_off_values(3.14159, 1)
    assert result == 3.1


def test_round_to_two_decimals():
    """Test rounding to two decimal places."""
    result = round_off_values(3.14159, 2)
    assert result == 3.14


def test_round_to_five_decimals():
    """Test rounding to five decimal places."""
    result = round_off_values(3.14159265359, 5)
    assert result == 3.14159


def test_round_negative_number():
    """Test rounding negative numbers."""
    result = round_off_values(-3.14159, 2)
    assert result == -3.14


def test_round_zero():
    """Test rounding zero."""
    result = round_off_values(0.0, 2)
    assert result == 0.0


def test_round_already_rounded():
    """Test rounding a number that's already at desired precision."""
    result = round_off_values(5.5, 1)
    assert result == 5.5


def test_round_integer_as_float():
    """Test rounding when input is effectively an integer."""
    result = round_off_values(42.0, 2)
    assert result == 42.0


def test_round_up_at_midpoint():
    """Test Python's banker's rounding (round half to even)."""
    result = round_off_values(2.5, 0)
    assert result == 2.0  # Python rounds to nearest even

    result = round_off_values(3.5, 0)
    assert result == 4.0  # Python rounds to nearest even


def test_round_very_small_number():
    """Test rounding very small numbers."""
    result = round_off_values(0.000123456, 6)
    assert result == 0.000123


def test_round_very_large_number():
    """Test rounding very large numbers."""
    result = round_off_values(123456789.987654321, 2)
    assert result == 123456789.99


def test_round_with_negative_decimals():
    """Test rounding to negative decimal places (tens, hundreds, etc)."""
    result = round_off_values(12345.67, -1)
    assert result == 12350.0

    result = round_off_values(12345.67, -2)
    assert result == 12300.0


# get_duration
def test_duration_in_days_one_day():
    """Test duration of exactly one day."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 2, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 1.0


def test_duration_in_days_multiple_days():
    """Test duration of multiple days."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 10, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 9.0


def test_duration_in_days_with_hours():
    """Test duration with partial days (hours)."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 12, 0, 0)  # 12 hours = 0.5 days
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 0.5


def test_duration_in_days_with_seconds():
    """Test duration with seconds included."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 2, 0, 0, 30)  # 1 day + 30 seconds
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    # 1 day = 86400 seconds, 30 seconds = 30/86400 ≈ 0.000347
    expected = round(1 + 30 / 86400, 2)
    assert result == expected


def test_duration_in_days_less_than_one_day():
    """Test duration less than one day."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 6, 0, 0)  # 6 hours
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 0.25  # 6 hours = 0.25 days


def test_duration_in_days_zero():
    """Test zero duration."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 0.0


def test_duration_in_days_across_month_boundary():
    """Test duration across month boundary."""
    since = datetime(2024, 1, 30, 0, 0, 0)
    until = datetime(2024, 2, 2, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 3.0  # Jan 30 -> Jan 31 -> Feb 1 -> Feb 2


def test_duration_in_days_across_year_boundary():
    """Test duration across year boundary."""
    since = datetime(2023, 12, 30, 0, 0, 0)
    until = datetime(2024, 1, 2, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 3.0


def test_duration_in_days_leap_year():
    """Test duration including leap day."""
    since = datetime(2024, 2, 28, 0, 0, 0)
    until = datetime(2024, 3, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    assert result == 2.0  # Feb 28 -> Feb 29 (leap day) -> Mar 1


def test_duration_in_months_one_month():
    """Test duration of exactly one month."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 2, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 1.0


def test_duration_in_months_multiple_months():
    """Test duration of multiple months."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 4, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 3.0


def test_duration_in_months_one_year():
    """Test duration of exactly one year."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2025, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 12.0


def test_duration_in_months_with_days():
    """Test duration with partial months (days)."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 16, 0, 0, 0)  # 15 days
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    # 15 days / 30.44 ≈ 0.49 months
    expected = round(15 / 30.44, 2)
    assert result == expected


def test_duration_in_months_partial():
    """Test duration of one month plus some days."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 2, 15, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    # 1 month + 14 days
    expected = round(1 + 14 / 30.44, 2)
    assert result == expected


def test_duration_in_months_zero():
    """Test zero duration in months."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 0.0


def test_duration_in_months_across_year():
    """Test duration spanning multiple years."""
    since = datetime(2023, 11, 1, 0, 0, 0)
    until = datetime(2024, 2, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 3.0  # Nov, Dec, Jan


def test_duration_in_months_multiple_years():
    """Test duration of multiple years."""
    since = datetime(2022, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 24.0  # 2 years = 24 months


def test_duration_in_months_leap_year_february():
    """Test duration through February in leap year."""
    since = datetime(2024, 2, 1, 0, 0, 0)
    until = datetime(2024, 3, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 1.0


def test_duration_invalid_unit():
    """Test error with invalid time unit."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 2, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    with pytest.raises(ValueError, match="time_unit must be either 'days' or 'months'"):
        get_duration(time_range, time_unit="years")


def test_duration_invalid_unit_weeks():
    """Test error with 'weeks' as time unit."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 8, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    with pytest.raises(ValueError, match="time_unit must be either 'days' or 'months'"):
        get_duration(time_range, time_unit="weeks")


def test_duration_default_unit_is_months():
    """Test that default time_unit is 'months'."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 2, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    # Call without specifying time_unit
    result = get_duration(time_range)
    assert result == 1.0  # Should default to months


def test_duration_rounding_precision():
    """Test that results are rounded to 2 decimal places."""
    since = datetime(2024, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 1, 23, 45)  # Complex time
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result_days = get_duration(time_range, time_unit="days")
    result_months = get_duration(time_range, time_unit="months")

    # Check that both have at most 2 decimal places
    assert len(str(result_days).split(".")[-1]) <= 2
    assert len(str(result_months).split(".")[-1]) <= 2


def test_duration_very_long_period_days():
    """Test duration over a very long period in days."""
    since = datetime(2020, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="days")
    # 4 years = 1461 days (including one leap year 2020)
    assert result == 1461.0


def test_duration_very_long_period_months():
    """Test duration over a very long period in months."""
    since = datetime(2020, 1, 1, 0, 0, 0)
    until = datetime(2024, 1, 1, 0, 0, 0)
    time_range = TimeRange(since=since, until=until, timezone=UTC_TIMEZONEINFO, time_format=DEFAULT_TIME_FORMAT)

    result = get_duration(time_range, time_unit="months")
    assert result == 48.0  # 4 years = 48 months
