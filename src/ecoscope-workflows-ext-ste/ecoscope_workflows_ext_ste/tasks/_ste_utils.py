import os
import ee
import hashlib
import pandas as pd
from pydantic import Field
from datetime import datetime
from ecoscope.trajectory import Trajectory
from typing import Annotated, Optional,Dict
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf
from ecoscope.analysis.seasons import seasonal_windows, std_ndvi_vals, val_cuts
from ecoscope_workflows_ext_ecoscope.tasks.io._earthengine import determine_season_windows


@task
def label_quarter_status(gdf: AnyGeoDataFrame, timestamp_col:str) -> AnyGeoDataFrame:
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

    gdf["quarter_status"] =gdf[timestamp_col].apply(
        lambda x: "present quarter" if x.to_period("Q") == present_quarter else "previous quarter"
    )
    return gdf

@task
def generate_speed_raster(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data")],
    dist_col: Annotated[str, Field(description="Column name for step distance")],
    output_dir: Annotated[str, Field(description="Directory to save the output raster")],
    filename: Annotated[
        Optional[str],
        Field(
            description="Filename for the output GeoTIFF (without extension). "
                        "If not provided, a hash of the data will be used.",
            exclude=True,
        )
    ] = None,
    resolution: Annotated[
        Optional[float], 
        Field(default=None, description="Resolution of the raster. If None, mean of dist_col is used.")
    ] = None,
    radius: Annotated[int, Field(default=2, description="Radius for kernel smoothing")] = 2,
    cutoff: Annotated[Optional[float], Field(default=None, description="Cutoff distance for kernel")] = None,
    tortuosity_length: Annotated[int, Field(default=3, description="Length scale for tortuosity smoothing")] = 3
) -> str:
    """
    Generate a mean interpolated speed raster from trajectory data.

    Returns:
        str: Path to the generated GeoTIFF raster file.
    """
    # If no filename is provided, generate it from a hash of the GeoDataFrame
    if not filename:
        df_hash = hashlib.sha256(pd.util.hash_pandas_object(gdf, index=True).values).hexdigest()
        filename = df_hash[:7]
        print(f"No filename provided. Generated filename: {filename}")

    mean_step_length = gdf[dist_col].mean()
    print(f"Mean step length: {mean_step_length}")
    res = resolution if resolution is not None else mean_step_length

    ecograph = Ecograph(
        Trajectory(gdf),
        resolution=res,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length
    )

    raster_path = os.path.join(output_dir, f"{filename}.tif")
    ecograph.to_geotiff("speed", raster_path, interpolation="mean")
    return raster_path

@task
def retrieve_feature_gdf(
    file_path: Annotated[str, Field(description="Path to the saved Ecograph feature file")]
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


def determine_season_windows(
    aoi: AnyGeoDataFrame, 
    since,
    until):
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
def create_seasonal_labels(
    project_name: Annotated[str, Field(description="GEE project to use")],
    traj: AnyGeoDataFrame,
    total_percentiles: AnyGeoDataFrame
) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.
    """
    try:
        SERVICE_ACCOUNT = os.getenv("EE_SERVICE_ACCOUNT")
        PRIVATE_KEY_FILE = os.getenv("EE_PRIVATE_KEY_FILE")
        if SERVICE_ACCOUNT and PRIVATE_KEY_FILE:
            credentials = ee.ServiceAccountCredentials(
                email=SERVICE_ACCOUNT,
                key_file=PRIVATE_KEY_FILE,
            )
            ee.Initialize(credentials)
        else:
            print(f"Initializing GEE client for project {project_name}")        
            print(f"Initializing earth-engine connection for {project_name} project.")
            ee.Authenticate()
            ee.Initialize(project=project_name)
            print("Successfully connected to EarthEngine.")
    except ee.EEException as e:
        print(f"Failed to connect to EarthEngine: {e}")
    
    try:
        print("Calculating seasonal ETD percentiles for entire trajectory")

        # Assume a single subject/area and get AOI by the only index or first row
        if len(total_percentiles) != 1:
            print("Expected total_percentiles to contain a single AOI row.")
            return None

        aoi = total_percentiles.iloc[0]

        seasonal_wins = determine_season_windows(
            aoi=aoi,
            since=traj["segment_start"].min(),
            until=traj["segment_end"].max(),
        )

        if seasonal_wins is None:
            print("No seasonal windows were determined.")
            traj["season"] = None
            return traj

        season_bins = pd.IntervalIndex(
            seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1)
        )
        labels = seasonal_wins["season"]

        traj["season"] = pd.cut(traj["segment_start"], bins=season_bins).map(
            dict(zip(season_bins, labels))
        )

        return traj

    except Exception as e:
        print(f"Failed to apply seasonal label to trajectory: {e}")
        return None

@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")]
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to split.
        column (str): The column to split by.

    Returns:
        Dict[str, gpd.GeoDataFrame]: Dictionary where keys are unique values of the column, and values are GeoDataFrames.
    """
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped