import os
import hashlib
import pandas as pd
import geopandas as gpd
from ecoscope.trajectory import Trajectory
from pydantic import Field, BaseModel, ConfigDict
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from typing import Annotated, Optional, Dict, cast, Literal
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf
from ecoscope.analysis.seasons import seasonal_windows, std_ndvi_vals, val_cuts
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame, AdvancedField
from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_elliptical_time_density


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
def label_quarter_status(gdf: AnyDataFrame, timestamp_col: str) -> AnyDataFrame:
    """
    Adds a 'quarter_status' column to the DataFrame based on whether each timestamp
    falls in the most recent quarter (from the dataset) or any previous quarter.

    Args:
        gdf (AnyDataFrame): GeoDataFrame with a datetime column.
        timestamp_col (str): Name of the datetime column to evaluate.

    Returns:
        AnyDataFrame: Updated GeoDataFrame with 'quarter_status' column added.
    """
    gdf = gdf.copy()
    gdf[timestamp_col] = pd.to_datetime(gdf[timestamp_col])

    latest_date = gdf[timestamp_col].max()
    most_recent_quarter = latest_date.to_period("Q")

    gdf["quarter_status"] = (
        gdf[timestamp_col]
        .dt.to_period("Q")
        .apply(lambda x: "Present Quarter Movement" if x == most_recent_quarter else "Previous Quarter Movement")
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
    step_length: Annotated[Optional[int], Field(default=None, description="Mean step length for resolution")] = None,
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
    if step_length is not None:
        print(f"Using provided mean step length: {step_length}")
        resolution = float(step_length)
    else:
        print("No step length provided, calculating from step distances.")
        step_length = float(dist_series.mean())
        print(f"Mean step length: {step_length}")
    res = float(resolution) if resolution is not None else step_length
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
            print(f"Failed to calculate ETD : {group_df.name if hasattr(group_df, 'name') else 'unknown'}: {e}")
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
def mapbook_info(values: Dict) -> Dict:
    val_dict = {}
    return val_dict.update(values)


@task
def generate_mcp_gdf(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",  # Africa Albers Equal Area
) -> AnyGeoDataFrame:
    """
    Create a Minimum Convex Polygon (MCP) from input point geometries and compute its area.
    """
    if gdf is None or gdf.empty:
        raise ValueError("Input GeoDataFrame is empty.")
    if gdf.geometry is None:
        raise ValueError("Input GeoDataFrame has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs

    # Filter out empty or null geometries
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("No valid geometries in input GeoDataFrame.")

    # Ensure only points (fallback: centroid of non-point geometries)
    if not all(valid_points_gdf.geometry.geom_type.isin(["Point"])):
        valid_points_gdf.geometry = valid_points_gdf.geometry.centroid

    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {"area_m2": [area_sq_meters], "area_km2": [area_sq_km], "mcp": "mcp"},
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf


ColumnName = Annotated[str, Field(description="Column to aggregate")]


@task
def dataframe_column_first_unique_str(
    df: AnyDataFrame,
    column_name: ColumnName,
) -> Annotated[str, Field(description="The first unique string value in the column")]:
    return str(df[column_name].unique()[0])
