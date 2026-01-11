import pytest
import pandas as pd
import geopandas as gpd
from shapely import Point
from ecoscope_workflows_ext_ste.tasks._merge import merge_multiple_df


@pytest.fixture
def sample_dfs():
    """Fixture to create sample DataFrames for testing."""
    df1 = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["Nairobi", "Mombasa"]})

    df2 = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40], "city": ["Kisumu", "Nakuru"]})

    df3 = pd.DataFrame({"name": ["Eve"], "age": [28], "city": ["Eldoret"]})

    return [df1, df2, df3]


@pytest.fixture
def sample_gdfs():
    """Fixture to create sample GeoDataFrames for testing."""
    gdf1 = gpd.GeoDataFrame(
        {"id": [1, 2], "name": ["Point A", "Point B"], "geometry": [Point(35.0, 1.0), Point(36.0, 1.0)]},
        crs="EPSG:4326",
    )

    gdf2 = gpd.GeoDataFrame(
        {"id": [3, 4], "name": ["Point C", "Point D"], "geometry": [Point(37.0, 1.0), Point(38.0, 1.0)]},
        crs="EPSG:4326",
    )

    return [gdf1, gdf2]


@pytest.fixture
def dfs_with_different_columns():
    """Fixture for DataFrames with different column sets."""
    df1 = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

    df2 = pd.DataFrame({"name": ["Charlie"], "city": ["Kisumu"]})

    return [df1, df2]


@pytest.fixture
def dfs_with_index():
    """Fixture for DataFrames with specific indices."""
    df1 = pd.DataFrame({"value": [10, 20]}, index=[0, 1])

    df2 = pd.DataFrame({"value": [30, 40]}, index=[0, 1])

    return [df1, df2]


def test_merge_two_dataframes(sample_dfs):
    """Test merging two DataFrames."""
    result = merge_multiple_df(sample_dfs[:2])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4  # 2 + 2
    assert list(result.columns) == ["name", "age", "city"]
    assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David"]


def test_merge_three_dataframes(sample_dfs):
    """Test merging three DataFrames."""
    result = merge_multiple_df(sample_dfs)

    assert len(result) == 5  # 2 + 2 + 1
    assert result["name"].tolist() == ["Alice", "Bob", "Charlie", "David", "Eve"]


def test_merge_single_dataframe(sample_dfs):
    """Test merging a single DataFrame (edge case)."""
    result = merge_multiple_df([sample_dfs[0]])

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert result.equals(sample_dfs[0])


def test_merge_geodataframes(sample_gdfs):
    """Test merging GeoDataFrames."""
    result = merge_multiple_df(sample_gdfs)

    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 4
    assert result.crs == sample_gdfs[0].crs
    assert all(result.geometry.geom_type == "Point")


# ignore_index parameter tests
def test_merge_ignore_index_true(dfs_with_index):
    """Test merge with ignore_index=True (default)."""
    result = merge_multiple_df(dfs_with_index, ignore_index=True)

    # Should create new sequential index
    assert result.index.tolist() == [0, 1, 2, 3]
    assert result["value"].tolist() == [10, 20, 30, 40]


def test_merge_ignore_index_false(dfs_with_index):
    """Test merge with ignore_index=False."""
    result = merge_multiple_df(dfs_with_index, ignore_index=False)

    # Should preserve original indices (which will be duplicated)
    assert result.index.tolist() == [0, 1, 0, 1]
    assert result["value"].tolist() == [10, 20, 30, 40]


# sort parameter tests
def test_merge_sort_false(dfs_with_different_columns):
    """Test merge with sort=False (default)."""
    result = merge_multiple_df(dfs_with_different_columns, sort=False)

    assert "name" in result.columns
    assert "age" in result.columns
    assert "city" in result.columns
    # Check for NaN in missing values
    assert pd.isna(result.loc[2, "age"])
    assert pd.isna(result.loc[0, "city"])


def test_merge_sort_true(dfs_with_different_columns):
    """Test merge with sort=True."""
    result = merge_multiple_df(dfs_with_different_columns, sort=True)

    # Columns should be sorted alphabetically
    assert list(result.columns) == sorted(["name", "age", "city"])


# Different column scenarios
def test_merge_different_columns_creates_nan(dfs_with_different_columns):
    """Test that missing columns are filled with NaN."""
    result = merge_multiple_df(dfs_with_different_columns)

    # df1 has no 'city', df2 has no 'age'
    assert pd.isna(result.loc[0, "city"])
    assert pd.isna(result.loc[1, "city"])
    assert pd.isna(result.loc[2, "age"])


def test_merge_completely_different_columns():
    """Test merging DataFrames with no common columns."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [3, 4]})

    result = merge_multiple_df([df1, df2])

    assert "a" in result.columns
    assert "b" in result.columns
    assert len(result) == 4
    assert pd.isna(result.loc[0, "b"])
    assert pd.isna(result.loc[2, "a"])


def test_merge_overlapping_columns():
    """Test merging DataFrames with some overlapping columns."""
    df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pd.DataFrame({"b": [5, 6], "c": [7, 8]})

    result = merge_multiple_df([df1, df2])

    assert list(result.columns) == ["a", "b", "c"]
    assert result["b"].tolist() == [3, 4, 5, 6]


# Different data types
def test_merge_mixed_dtypes():
    """Test merging DataFrames with mixed data types."""
    df1 = pd.DataFrame({"int_col": [1, 2], "str_col": ["a", "b"], "float_col": [1.1, 2.2]})

    df2 = pd.DataFrame({"int_col": [3, 4], "str_col": ["c", "d"], "float_col": [3.3, 4.4]})

    result = merge_multiple_df([df1, df2])

    assert result["int_col"].dtype == df1["int_col"].dtype
    assert result["str_col"].dtype == df1["str_col"].dtype
    assert result["float_col"].dtype == df1["float_col"].dtype


def test_merge_with_datetime():
    """Test merging DataFrames with datetime columns."""
    df1 = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "value": [10, 20]})

    df2 = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04"]), "value": [30, 40]})

    result = merge_multiple_df([df1, df2])

    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert len(result) == 4


def test_merge_with_categorical():
    """Test merging DataFrames with categorical columns."""
    df1 = pd.DataFrame({"category": pd.Categorical(["A", "B"])})

    df2 = pd.DataFrame({"category": pd.Categorical(["C", "D"])})

    result = merge_multiple_df([df1, df2])

    assert len(result) == 4
    assert "category" in result.columns


# Empty and edge cases
def test_merge_empty_list_raises_error():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="list_df cannot be empty"):
        merge_multiple_df([])


def test_merge_with_empty_dataframe():
    """Test merging when one DataFrame is empty."""
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": []})

    result = merge_multiple_df([df1, df2])

    assert len(result) == 2
    assert result["a"].tolist() == [1, 2]


def test_merge_all_empty_dataframes():
    """Test merging when all DataFrames are empty."""
    df1 = pd.DataFrame({"a": []})
    df2 = pd.DataFrame({"a": []})

    result = merge_multiple_df([df1, df2])

    assert len(result) == 0
    assert "a" in result.columns


def test_merge_large_number_of_dataframes():
    """Test merging many DataFrames."""
    dfs = [pd.DataFrame({"value": [i]}) for i in range(100)]

    result = merge_multiple_df(dfs)

    assert len(result) == 100
    assert result["value"].tolist() == list(range(100))


# Real-world scenarios
def test_merge_trajectory_dataframes():
    """Test merging trajectory DataFrames (realistic wildlife tracking)."""
    traj1 = pd.DataFrame(
        {
            "subject_name": ["Elephant1", "Elephant1"],
            "speed_kmhr": [5.2, 6.1],
            "duration_status": ["Current", "Current"],
        }
    )

    traj2 = pd.DataFrame(
        {
            "subject_name": ["Elephant1", "Elephant1"],
            "speed_kmhr": [4.8, 5.5],
            "duration_status": ["Previous", "Previous"],
        }
    )

    result = merge_multiple_df([traj1, traj2])

    assert len(result) == 4
    assert result["duration_status"].value_counts()["Current"] == 2
    assert result["duration_status"].value_counts()["Previous"] == 2


def test_merge_preserves_column_order():
    """Test that column order is preserved from first DataFrame."""
    df1 = pd.DataFrame({"c": [1], "b": [2], "a": [3]})
    df2 = pd.DataFrame({"c": [4], "b": [5], "a": [6]})

    result = merge_multiple_df([df1, df2], sort=False)

    assert list(result.columns) == ["c", "b", "a"]


def test_merge_with_duplicate_rows():
    """Test merging DataFrames with duplicate rows."""
    df1 = pd.DataFrame({"a": [1, 1, 2]})
    df2 = pd.DataFrame({"a": [1, 2, 2]})

    result = merge_multiple_df([df1, df2])

    assert len(result) == 6
    assert result["a"].value_counts()[1] == 3
    assert result["a"].value_counts()[2] == 3


# Type preservation tests
def test_merge_returns_geodataframe_when_input_is_geodataframe(sample_gdfs):
    """Test that merging GeoDataFrames returns a GeoDataFrame."""
    result = merge_multiple_df(sample_gdfs)

    assert isinstance(result, gpd.GeoDataFrame)
    assert hasattr(result, "geometry")


def test_merge_returns_dataframe_when_input_is_dataframe(sample_dfs):
    """Test that merging DataFrames returns a DataFrame."""
    result = merge_multiple_df(sample_dfs)

    assert isinstance(result, pd.DataFrame)
    assert not isinstance(result, gpd.GeoDataFrame)


def test_merge_mixed_df_and_gdf(sample_dfs, sample_gdfs):
    """Test merging mix of DataFrame and GeoDataFrame."""
    # Note: This might fail or produce unexpected results
    # depending on pandas/geopandas behavior
    result = merge_multiple_df([sample_dfs[0], sample_gdfs[0]])

    # Result type depends on concat behavior
    assert isinstance(result, (pd.DataFrame, gpd.GeoDataFrame))
    assert len(result) == 4
