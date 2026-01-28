import geopandas as gpd
from pydantic import Field
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Dict, Literal, Union, List
from ecoscope_workflows_core.tasks.filter._filter import TimeRange
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame


@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`split_gdf_by_column`:gdf is empty.")

    if column not in gdf.columns:
        raise ValueError(f"`split_gdf_by_column`:Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped


@task
def generate_mcp_gdf(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",  # Africa Albers Equal Area
) -> AnyGeoDataFrame:
    """
    Create a Minimum Convex Polygon (MCP) from input point geometries and compute its area.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`generate_mcp_gdf`:gdf is empty.")
    if gdf.geometry is None:
        raise ValueError("`generate_mcp_gdf`:gdf has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("`generate_mcp_gdf`:gdf must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("`generate_mcp_gdf`:No valid geometries in gdf.")

    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    if not all(projected_gdf.geometry.geom_type.isin(["Point"])):
        projected_gdf.geometry = projected_gdf.geometry.centroid

    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {"area_m2": [area_sq_meters], "area_km2": [area_sq_km]},
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf


@task
def round_off_values(value: float, dp: int) -> float:
    return round(value, dp)


@task
def dataframe_column_first_unique_str(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="Column to aggregate")],
) -> Annotated[str, Field(description="The first unique string value in the column (sentence case)")]:
    """
    Get the first unique string value from a column and convert to sentence case.

    Args:
        df: Input DataFrame
        column_name: Column name to extract value from

    Returns:
        First unique value from the column, converted to sentence case

    Raises:
        ValueError: If df is empty or column doesn't exist
    """
    if df is None or df.empty:
        raise ValueError("df is empty")

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    unique_values = df[column_name].unique()

    if len(unique_values) == 0:
        raise ValueError(f"No values found in column '{column_name}'")

    return str(unique_values[0]).capitalize()


@task
def get_duration(
    time_range: Annotated[
        TimeRange,
        Field(description="Time range object"),
    ],
    time_unit: Literal["days", "months"] = "months",
) -> float:
    """
    Compute the duration of a TimeRange object in days or months.

    Args:
        time_range (TimeRange): The TimeRange returned by set_time_range.
        time_unit (str): Either "days" or "months". Defaults to "months".

    Returns:
        float: Duration of the period in the requested unit.
    """
    since = time_range.since
    until = time_range.until

    if time_unit == "days":
        return round((until - since).days + (until - since).seconds / 86400.0, 2)

    elif time_unit == "months":
        from dateutil.relativedelta import relativedelta

        rd = relativedelta(until, since)
        months = rd.years * 12 + rd.months + rd.days / 30.44
        return round(months, 2)

    else:
        raise ValueError("`get_duration`:time_unit must be either 'days' or 'months'")


@task
def filter_df_cols(df: AnyDataFrame, columns: Union[str, List[str]]) -> AnyDataFrame:
    """
    Filter DataFrame to include only specified columns.

    Args:
        df: Input DataFrame
        columns: Column name(s) to keep

    Returns:
        DataFrame with only the specified columns

    Raises:
        ValueError: If any specified columns are not present in the DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]

    # Check for missing columns
    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}. " f"Available columns: {list(df.columns)}")

    return df[columns]


@task
def create_column(df: AnyDataFrame, col_name: str, value: int | float | str) -> AnyDataFrame:
    """
    Create a new column in the DataFrame with a default value if it doesn't already exist.

    Args:
        df: Input DataFrame
        col_name: Name of the column to create
        default_value: Default value to assign to the new column

    Returns:
        DataFrame with the new column added (if it was missing)
    """
    df = df.copy()
    if col_name not in df.columns:
        df[col_name] = value
    return df


@task
def convert_to_str(
    df: AnyDataFrame,
    columns: Union[str, List[str]],
) -> AnyDataFrame:
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
            continue

        try:
            df[column] = df[column].astype(str)
        except Exception as e:
            print(f"Error converting column '{column}' to int: {e}")

    return df
