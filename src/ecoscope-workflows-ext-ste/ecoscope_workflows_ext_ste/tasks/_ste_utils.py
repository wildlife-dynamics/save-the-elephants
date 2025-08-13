import os
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from shapely.geometry import Polygon
from ecoscope.trajectory import Trajectory
from pydantic.json_schema import SkipJsonSchema
from pydantic import Field, BaseModel, ConfigDict
from ecoscope_workflows_core.decorators import task
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf
from typing import Annotated, Optional, Dict, cast, Literal, Union, List, Tuple, Callable
from ecoscope.analysis.seasons import seasonal_windows, std_ndvi_vals, val_cuts
from ecoscope_workflows_ext_ecoscope.tasks.results._ecoplot import ExportArgs
from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_elliptical_time_density

from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame, AdvancedField


class AutoScaleGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Auto-scale"})
    auto_scale_or_custom: Annotated[
        Literal["Auto-scale"],
        AdvancedField(
            default="Auto-scale",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Auto-scale"


class CustomGridCellSize(BaseModel):
    model_config = ConfigDict(json_schema_extra={"title": "Customize"})
    auto_scale_or_custom: Annotated[
        Literal["Customize"],
        AdvancedField(
            default="Customize",
            title=" ",
            description="Define the resolution of the raster grid (in meters per pixel).",
        ),
    ] = "Customize"
    grid_cell_size: Annotated[
        float | SkipJsonSchema[None],
        Field(
            description="Custom Raster Pixel Size (Meters)",
            gt=0,
            lt=10000,
            default=5000,
            json_schema_extra={"exclusiveMinimum": 0, "exclusiveMaximum": 10000},
        ),
    ] = 5000


@task
def label_quarter_status(gdf: AnyGeoDataFrame, timestamp_col: str) -> AnyGeoDataFrame:
    """
    Adds a 'quarter_status' column to the DataFrame based on whether each timestamp
    falls in the most recent quarter or a previous one.

    Args:
        trajs (pd.DataFrame): DataFrame with a datetime column.
        timestamp_col (str): Name of the datetime column to evaluate.

    Returns:
        pd.DataFrame: Updated DataFrame with 'quarter_status' column added.
    """
    gdf[timestamp_col] = pd.to_datetime(gdf[timestamp_col])
    latest_date = gdf[timestamp_col].max()
    present_quarter = latest_date.to_period("Q")

    gdf["quarter_status"] = gdf[timestamp_col].apply(
        lambda x: "present quarter" if x.to_period("Q") == present_quarter else "previous quarter"
    )
    return gdf


@task
def generate_ecograph_raster(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data")],
    dist_col: Annotated[str, Field(description="Column name for step distance")],
    output_dir: Annotated[str, Field(description="Directory to save the output raster")],
    filename: Annotated[
        Optional[str],
        Field(
            description="Filename for the output GeoTIFF (without extension). "
            "If not provided, a hash of the data will be used.",
            exclude=True,
        ),
    ] = None,
    resolution: Annotated[
        Optional[float], Field(default=None, description="Raster resolution; if None, uses the mean of dist_col.")
    ] = None,
    radius: Annotated[int, Field(default=2, description="Radius for kernel smoothing")] = 2,
    cutoff: Annotated[Optional[float], Field(default=None, description="Cutoff distance for kernel")] = None,
    tortuosity_length: Annotated[int, Field(default=3, description="Length scale for tortuosity smoothing")] = 3,
    interpolation: Literal["mean", "min", "max", "median"] = "mean",
    # Choose exactly ONE of the following two:
    movement_covariate: Optional[
        Literal["dot_product", "step_length", "speed", "sin_time", "cos_time", "tortuosity_1", "tortuosity_2"]
    ] = None,
    network_metric: Optional[Literal["weight", "betweenness", "degree", "collective_influence"]] = None,
) -> str:
    """
    Generate a GeoTIFF raster from trajectory data using Ecograph.

    Exactly one of `movement_covariate` or `network_metric` must be provided.
    If `resolution` is None, the mean of `dist_col` is used.
    """

    # ---- validation ----
    if gdf is None or len(gdf) == 0:
        raise ValueError("gdf is empty.")

    if dist_col not in gdf.columns:
        raise ValueError(f"Column '{dist_col}' not found in gdf.")

    # ensure numeric (coerce errors to NaN then drop before mean)
    dist_series = pd.to_numeric(gdf[dist_col], errors="coerce")
    if dist_series.dropna().empty:
        raise ValueError(f"Column '{dist_col}' has no numeric values to compute a mean resolution.")

    if (movement_covariate is None) == (network_metric is None):
        raise ValueError("Provide exactly one of 'movement_covariate' or 'network_metric'.")

    if not filename:
        # hash the dataframe contents (including geometry)
        df_hash = hashlib.sha256(pd.util.hash_pandas_object(gdf, index=True).values).hexdigest()
        filename = df_hash[:7]
        print(f"No filename provided. Generated filename: {filename}")

    mean_step_length = float(dist_series.mean())
    print(f"Mean step length: {mean_step_length}")
    res = float(resolution) if resolution is not None else mean_step_length
    if res <= 0:
        raise ValueError(f"Computed/Provided resolution must be > 0, got {res}.")

    os.makedirs(output_dir, exist_ok=True)
    raster_path = os.path.join(output_dir, f"{filename}.tif")

    ecograph = Ecograph(
        Trajectory(gdf),
        resolution=res,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length,
    )

    covariate = movement_covariate if movement_covariate is not None else network_metric
    ecograph.to_geotiff(covariate, raster_path, interpolation=interpolation)

    return raster_path


@task
def retrieve_feature_gdf(
    file_path: Annotated[str, Field(description="Path to the saved Ecograph feature file")],
) -> AnyGeoDataFrame:
    """
    Loads a GeoDataFrame from a saved Ecograph feature file.
    Args:
        file_path (str): Path to the `.geojson` or `.gpkg` feature file.

    Returns:
        AnyGeoDataFrame: The loaded GeoDataFrame containing spatial features.
    """
    gdf = get_feature_gdf(file_path)
    return gdf


def determine_season_windows(aoi: AnyGeoDataFrame, since, until):
    windows = None
    try:
        # Merge to a larger Polygon
        aoi = aoi.copy()
        aoi = aoi.to_crs(4326)
        aoi = aoi.dissolve()
        aoi = aoi.iloc[0]["geometry"]

        # Determine wet/dry seasons
        print(f"Attempting download of NDVI values since: {since.isoformat()} until: {until.isoformat()}")
        date_chunks = (
            pd.date_range(start=since, end=until, periods=5, inclusive="both")
            .to_series()
            .apply(lambda x: x.isoformat())
            .values
        )
        ndvi_vals = []
        for t in range(1, len(date_chunks)):
            print(f"Downloading NDVI Values from EarthEngine......({t}/5)")
            ndvi_vals.append(
                std_ndvi_vals(
                    img_coll="MODIS/061/MCD43A4",
                    nir_band="Nadir_Reflectance_Band2",
                    red_band="Nadir_Reflectance_Band1",
                    aoi=aoi,
                    start=date_chunks[t - 1],
                    end=date_chunks[t],
                )
            )
        ndvi_vals = pd.concat(ndvi_vals)

        print(f"Calculating seasonal windows based on {str(len(ndvi_vals))} NDVI values....")

        # Calculate the seasonal transition point
        cuts = val_cuts(ndvi_vals, 2)

        # Determine the seasonal time windows
        windows = seasonal_windows(ndvi_vals, cuts, season_labels=["dry", "wet"])

    except Exception as e:
        print(f"Failed to calculate seasonal windows {e}")

    return windows


@task
def create_seasonal_labels(traj: AnyGeoDataFrame, total_percentiles: AnyDataFrame) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.
    """
    try:
        print("Calculating seasonal ETD percentiles for entire trajectory")
        print(f"Total percentiles shape: {total_percentiles.shape}")
        print(f"Available seasons: {total_percentiles['season'].unique()}")

        # Since total_percentiles contains the seasonal windows directly,
        # we don't need determine_season_windows() - we can use it directly
        seasonal_wins = total_percentiles.copy()

        # Filter to trajectory time range if needed
        traj_start = traj["segment_start"].min()
        traj_end = traj["segment_end"].max()

        # Keep only seasonal windows that overlap with trajectory timeframe
        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        print(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        print(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            print("No seasonal windows overlap with trajectory timeframe.")
            traj["season"] = None
            return traj

        # Create interval index
        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        print(f"Created {len(season_bins)} seasonal bins")

        labels = seasonal_wins["season"].values

        # Use pd.cut to assign segments to seasonal bins
        traj["season"] = pd.cut(traj["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )

        # Handle segments that fall outside seasonal windows
        null_count = traj["season"].isnull().sum()
        if null_count > 0:
            print(f"Warning: {null_count} trajectory segments couldn't be assigned to any season")

        print("Seasonal labeling complete. Season distribution:")
        print(traj["season"].value_counts(dropna=False))

        return traj

    except Exception as e:
        print(f"Failed to apply seasonal label to trajectory: {e}")
        import traceback

        traceback.print_exc()
        return None


@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to split.
        column (str): The column to split by.

    Returns:
        Dict[str, gpd.GeoDataFrame]: Dictionary where keys are unique values of the column are GeoDataFrames.
    """
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped


@task
def calculate_etd_by_groups(
    trajectory_gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="The trajectory geodataframe.", exclude=True),
    ],
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'extra__name'])",
            json_schema_extra={"default": ["groupby_col", "extra__name"]},
        ),
    ] = None,
    auto_scale_or_custom_cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
        Field(
            json_schema_extra={
                "title": "Auto Scale Or Custom Grid Cell Size",
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom": "Auto-scale"},
            },
        ),
    ] = None,
    crs: Annotated[
        str,
        AdvancedField(
            default="EPSG:3857",
            title="Coordinate Reference System",
            description="The coordinate reference system in which to perform the density calculation",
        ),
    ] = "EPSG:3857",
    nodata_value: Annotated[float | str, AdvancedField(default="nan")] = "nan",
    band_count: Annotated[int, AdvancedField(default=1)] = 1,
    max_speed_factor: Annotated[
        float,
        AdvancedField(
            default=1.05,
            title="Max Speed Factor (Kilometers per Hour)",
            description="An estimate of the subject's maximum speed.",
        ),
    ] = 1.05,
    expansion_factor: Annotated[
        float,
        AdvancedField(
            default=1.3,
            title="Shape Buffer Expansion Factor",
            description="Controls how far time density values spread across the grid.",
        ),
    ] = 1.3,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = None,
    include_groups: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to include grouping columns in the result",
        ),
    ] = False,
) -> AnyDataFrame:
    """
    Calculate Elliptical Time Density (ETD) for trajectory groups.

    This function applies calculate_elliptical_time_density to each group
    defined by the groupby_cols, similar to:

    trajs.groupby(["groupby_col", "extra__name"]).apply(
        lambda df: calculate_elliptical_time_density(df, ...),
        include_groups=False,
    )

    Args:
        trajectory_gdf: The trajectory geodataframe
        groupby_cols: List of column names to group by
        **kwargs: All other parameters passed to calculate_elliptical_time_density

    Returns:
        DataFrame with ETD results for all groups combined
    """

    # Set default groupby columns if not provided
    if groupby_cols is None:
        groupby_cols = ["groupby_col", "extra__name"]

    # Set default percentiles if not provided
    if percentiles is None:
        percentiles = [25.0, 50.0, 75.0, 90.0, 95.0, 99.9]

    # Validate that groupby columns exist
    missing_cols = [col for col in groupby_cols if col not in trajectory_gdf.columns]
    if missing_cols:
        raise ValueError(f"Groupby columns {missing_cols} not found in trajectory_gdf")

    def apply_etd_to_group(group_df):
        """Apply calculate_elliptical_time_density to a single group"""
        try:
            result = calculate_elliptical_time_density(
                trajectory_gdf=group_df,
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                crs=crs,
                nodata_value=nodata_value,
                band_count=band_count,
                max_speed_factor=max_speed_factor,
                expansion_factor=expansion_factor,
                percentiles=percentiles,
            )
            return result
        except Exception as e:
            print(f"Failed to calculate ETD for group {group_df.name if hasattr(group_df, 'name') else 'unknown'}: {e}")
            # Return empty DataFrame with correct schema
            return pd.DataFrame(
                {
                    "percentile": pd.Series(dtype="float64"),
                    "geometry": gpd.GeoSeries(dtype="geometry"),
                    "area_sqkm": pd.Series(dtype="float64"),
                }
            )

    # Apply ETD calculation to each group
    try:
        grouped_results = trajectory_gdf.groupby(groupby_cols).apply(apply_etd_to_group, include_groups=include_groups)

        # Reset index to get a clean DataFrame
        if include_groups:
            result = grouped_results.reset_index()
        else:
            result = grouped_results.reset_index(level=groupby_cols, drop=True).reset_index(drop=True)

        return cast(AnyDataFrame, result)

    except Exception as e:
        print(f"Failed to calculate ETD by groups: {e}")
        empty_result = pd.DataFrame(
            {
                "percentile": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(dtype="geometry"),
                "area_sqkm": pd.Series(dtype="float64"),
            }
        )
        return cast(AnyDataFrame, empty_result)


@task
def filter_column_values(
    df: AnyGeoDataFrame, column: str, values: Union[int, float, str, List[Union[int, float, str]]], include: bool = True
) -> AnyGeoDataFrame:
    """
    Filter DataFrame rows based on values in a specified column.

    Args:
        df: Input DataFrame/GeoDataFrame
        column: Column name to filter on
        values: Single value or list of values (int, float, or str) to filter by
        include: If True, keep rows with these values; if False, exclude them

    Returns:
        Filtered DataFrame/GeoDataFrame
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
        return df

    # Normalize to list
    if not isinstance(values, list):
        values = [values]

    # Filter based on include/exclude logic
    if include:
        filtered_df = df[df[column].isin(values)]
    else:
        filtered_df = df[~df[column].isin(values)]

    print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
    return filtered_df


def hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """
    Convert a hex color code to an RGBA tuple.

    Args:
        hex_color (str): Hex color string, e.g. '#bd7ebe'.
        alpha (int): Alpha value (0–255). Default is 255 (fully opaque).

    Returns:
        tuple: (R, G, B, A)
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 characters long.")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b, alpha)


@task
def convert_col_to_rgba(df: AnyDataFrame, col: str, new_col: str) -> AnyDataFrame:
    """
    Converts a column of hex colors into RGBA format and stores the result in a specified column.

    Args:
        df (AnyDataFrame): Input DataFrame containing the color column.
        col (str): Name of the column containing hex color strings.
        new_col (str): Name of the new column to store RGBA values.

    Returns:
        AnyDataFrame: DataFrame with an added column containing RGBA tuples.
    """
    df[new_col] = df[col].apply(lambda hex_val: hex_to_rgba(hex_val, alpha=255))
    return df


@task
def add_day_night_column(df: AnyGeoDataFrame, source_col: str, new_col: str = "day_night") -> AnyGeoDataFrame:
    """
    Adds a day/night classification column to the DataFrame based on a boolean column.

    Args:
        df (AnyGeoDataFrame): Input DataFrame/GeoDataFrame.
        source_col (str): Name of the boolean column to check.
        new_col (str): Name of the new column to store the day/night classification (default: 'day_night').

    Returns:
        AnyGeoDataFrame: DataFrame with an added classification column.

    Raises:
        KeyError: If source_col is missing from the DataFrame.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame.")

    df[new_col] = np.where(df[source_col], "Night", "Day")
    return df


@task
def compute_fix_density_and_night_class(
    selection: AnyGeoDataFrame,
    grid: AnyGeoDataFrame,
    threshold: float = 0.5,
    geometry_type: Literal["point", "line"] = "point",
    mode: Literal["proportion", "dominance"] = "proportion",
) -> AnyGeoDataFrame:
    # --- Ensure matching CRS (grid is 3857, selection is 4326) ---
    if getattr(selection, "crs", None) is None or getattr(grid, "crs", None) is None:
        raise ValueError("Both 'selection' and 'grid' must have a CRS set.")
    if selection.crs != grid.crs:
        selection = selection.to_crs(grid.crs)

    def classify_cell(cell_geom: Polygon) -> Tuple[Optional[float], Optional[str]]:
        if geometry_type == "point":
            # NOTE: 'within' excludes boundary; use covers(cell_geom) if you want boundary included
            points_in_cell = selection[selection.geometry.within(cell_geom)]

            if points_in_cell.empty:
                return np.nan, None

            night_ratio = points_in_cell["is_night"].mean()

            if mode == "proportion":
                if night_ratio == 0:
                    return len(points_in_cell), None
                elif night_ratio < threshold:
                    return len(points_in_cell), f"0.0–{threshold}"
                else:
                    return len(points_in_cell), f"{threshold}–1.0"
            elif mode == "dominance":
                return (len(points_in_cell), "night" if night_ratio >= threshold else "day")
            else:
                raise ValueError("Unsupported mode. Use 'proportion' or 'dominance'.")

        elif geometry_type == "line":
            lines_in_cell = selection[selection.geometry.intersects(cell_geom)]

            if lines_in_cell.empty:
                return np.nan, None

            avg_night_ratio = lines_in_cell["is_night_ratio"].mean()

            if mode == "proportion":
                if avg_night_ratio == 0:
                    return len(lines_in_cell), None
                elif avg_night_ratio < threshold:
                    return len(lines_in_cell), f"0.0 – {threshold}"
                else:
                    return len(lines_in_cell), f"{threshold} – 1.0"
            elif mode == "dominance":
                raise NotImplementedError("Dominance mode is not supported for line geometry.")
            else:
                raise ValueError("Unsupported mode. Use 'proportion' or 'dominance'.")
        else:
            raise ValueError("Unsupported geometry type")

    results = grid.geometry.apply(classify_cell)
    grid["density"] = results.apply(lambda x: x[0])
    column_name = "night_class" if mode == "proportion" else "time_dominance"
    grid[column_name] = results.apply(lambda x: x[1])
    return grid


@task
def filter_time_dominance(df: AnyGeoDataFrame, valid_dominance: List[str]) -> AnyGeoDataFrame:
    """
    Filters rows where 'time_dominance' is in the provided valid_dominance list.

    Args:
        df (AnyGeoDataFrame): Input DataFrame/GeoDataFrame.
        valid_dominance (List[str]): List of valid dominance values.

    Returns:
        AnyGeoDataFrame: Filtered DataFrame containing only rows with valid 'time_dominance'.
    """
    if "time_dominance" not in df.columns:
        raise KeyError("Column 'time_dominance' not found in DataFrame.")

    return df[df["time_dominance"].isin(valid_dominance)]


@task
def drop_missing_values_by_column(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="The column to check")],
) -> AnyDataFrame:
    """
    Drops rows where the specified column has NaN, None, or empty string values.

    Works for both numeric and non-numeric columns.
    """
    mask = df[column_name].notna() & (df[column_name] != "")
    return cast(AnyDataFrame, df[mask])


@task
def spatial_join(
    trajs_gdf: AnyGeoDataFrame,
    local_gdf: AnyGeoDataFrame,
    how: Literal["left", "right", "inner"] = "left",
    predicate: Literal[
        "intersects", "contains", "within", "touches", "crosses", "overlaps", "covers", "covered_by", "dwithin"
    ] = "within",
    distance: float | None = None,
) -> AnyGeoDataFrame:
    """
    Performs a spatial join between points and polygons (or any two GeoDataFrames).

    Args:
        points_gdf: The left GeoDataFrame (e.g., points).
        polygons_gdf: The right GeoDataFrame (e.g., polygons).
        how: Join type ('left', 'right', 'inner').
        predicate: Spatial predicate to use (e.g., 'within', 'intersects', 'dwithin').
        distance: Optional distance in CRS units (required if predicate='dwithin').

    Returns:
        GeoDataFrame: Joined GeoDataFrame.
    """
    if trajs_gdf.crs != local_gdf.crs:
        local_gdf = local_gdf.to_crs(trajs_gdf.crs)

    join_kwargs = {"how": how, "predicate": predicate}
    if predicate == "dwithin":
        if distance is None:
            raise ValueError("distance must be specified when using predicate='dwithin'")
        join_kwargs["distance"] = distance

    return gpd.sjoin(trajs_gdf, local_gdf, **join_kwargs)


@task
def assign_column(
    df: AnyDataFrame,
    column_name: Annotated[str, Field(description="Name of column to create/overwrite")],
    value: Annotated[Union[str, int, float], Field(description="Value to assign to every row")],
) -> AnyDataFrame:
    out = df.copy()
    out[column_name] = value
    return cast(AnyDataFrame, out)


@task
def assign_value_by_index(
    df: Annotated[AnyGeoDataFrame, Field(description="Target DataFrame/GeoDataFrame to modify")],
    subset_df: Annotated[AnyGeoDataFrame, Field(description="Rows whose index marks positions to update")],
    column_name: Annotated[str, Field(description="Column to create/overwrite")],
    value: Annotated[Union[str, int, float], Field(description="Value to assign at matching indices")],
    create_if_missing: Annotated[bool, Field(description="Create column if it does not exist", default=True)] = True,
) -> AnyGeoDataFrame:
    """
    For all indices present in subset_df.index, set df.loc[index, column_name] = value.
    Creates the column if missing (unless create_if_missing=False).
    """
    out = df.copy()

    if column_name not in out.columns and not create_if_missing:
        raise ValueError(f"Column '{column_name}' does not exist and create_if_missing=False.")

    # Only update indices that exist in the target df
    idx = subset_df.index.intersection(out.index)

    if len(idx) == 0:
        print("No overlapping indices to update.")
        return cast(AnyGeoDataFrame, out)

    out.loc[idx, column_name] = value
    print(f"Updated {len(idx)} row(s) in '{column_name}'.")

    return cast(AnyGeoDataFrame, out)


@task
def category_summary(
    df: Annotated[AnyDataFrame, Field(description="Target DataFrame")],
    group_cols: List[str],
    category_col: str,
    values: Optional[str] = None,  # None → count rows; else aggregate this column
    agg: Union[str, Callable] = "size",  # e.g. "size", "sum", "mean", np.sum, etc.
    fill_value: Union[int, float] = 0,
    pct_for: Optional[List[str]] = None,  # which category columns to add % for; None → all
    pct_suffix: str = "_pct",
    round_pct: Optional[int] = 2,  # decimals to round % (None to skip rounding)
    reset_index: bool = False,  # return a flat frame if True
) -> AnyDataFrame:
    """
    Group by `group_cols`, spread `category_col` into columns, compute totals,
    and add percentage columns for selected categories.

    Example (your case):
    out = category_summary(
        trajs,
        group_cols=["extra__name", "season"],
        category_col="area_status",
        pct_for=["Protected", "Unprotected"]
    )
    """
    group_cols = list(group_cols)

    # 1) Aggregate long → wide
    if values is None or agg == "size":
        long = df.groupby(group_cols + [category_col]).size().rename("__val__")
    else:
        long = df.groupby(group_cols + [category_col])[values].agg(agg).rename("__val__")

    wide = long.unstack(category_col, fill_value=fill_value)

    # Normalize category column labels to strings so pct_for can be List[str]
    wide.columns = [str(c) for c in wide.columns]

    # 2) Total across category columns (all current columns are category columns)
    category_cols = list(wide.columns)
    wide["Total"] = wide[category_cols].sum(axis=1)

    # 3) Percent columns
    if pct_for is None:
        pct_for = category_cols

    total = wide["Total"].to_numpy()
    total_safe = np.where(total == 0, np.nan, total)  # avoid division by zero → NaN

    for cat in pct_for:
        if cat in wide.columns:
            pct = (wide[cat].to_numpy() / total_safe) * 100.0
            if round_pct is not None:
                pct = np.round(pct, round_pct)
            wide[f"{cat}{pct_suffix}"] = pct
        else:
            wide[f"{cat}{pct_suffix}"] = np.nan

    wide = wide.fillna(0)

    if reset_index:
        wide = wide.reset_index()

    return wide


# rewrite
@task
def plot_protected_fix_proportions(
    summary: AnyDataFrame,
    title: str = "Proportion of Fixes (Seasonal)",
) -> str:
    elephants = sorted(set(idx[0] for idx in summary.index))

    protected_vals = []
    unprotected_vals = []
    custom_xticks = []
    xtick_positions = []

    for i, elephant in enumerate(elephants):
        for j, season in enumerate(["dry", "wet"]):
            idx = (elephant, season)
            protected = summary.loc[idx]["protected_pct"] if idx in summary.index else 0
            unprotected = summary.loc[idx]["unprotected_pct"] if idx in summary.index else 0

            protected_vals.append(protected)
            unprotected_vals.append(unprotected)
            custom_xticks.append(f"{season.capitalize()}\n{elephant}")
            xtick_positions.append(i * 2 + j)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=xtick_positions,
            y=protected_vals,
            name="Protected",
            marker_color="seagreen",
            hovertemplate="Protected: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=xtick_positions,
            y=unprotected_vals,
            name="Unprotected",
            marker_color="darkorange",
            hovertemplate="Unprotected: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(
            tickmode="array",
            tickvals=xtick_positions,
            ticktext=custom_xticks,
            tickangle=0,
            tickfont=dict(size=11),
            title=None,
        ),
        yaxis=dict(
            title="Proportion of Fixes",
            range=[0, 100],
            ticksuffix="%",
            tickfont=dict(size=12),
        ),
        legend=dict(title="Area Status", orientation="h", x=0.5, xanchor="center", y=1.1),
        plot_bgcolor="white",
        margin=dict(l=60, r=20, t=80, b=100),
        width=1400,
        height=600,
        shapes=[
            dict(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="black", width=1),
                fillcolor="rgba(0,0,0,0)",
            )
        ],
    )

    return fig.to_html(**ExportArgs().model_dump())
