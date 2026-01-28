import os
import logging
import hashlib
import pandas as pd
from pydantic import Field
from ecoscope.trajectory import Trajectory
from typing import Annotated, Optional, Literal
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope.analysis.ecograph import Ecograph, get_feature_gdf
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme

logger = logging.getLogger(__name__)


@task
def generate_ecograph_raster(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data")],
    dist_col: Annotated[str, Field(description="Column name for step distance")],
    output_dir: Annotated[Optional[str], Field(description="Directory to save the output raster")] = None,
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
    logger.info("Starting generate_ecograph_raster task.")
    if gdf is None or gdf.empty:
        raise ValueError("`generate_ecograph_raster`:Trajectory gdf is empty.")

    if dist_col not in gdf.columns:
        raise ValueError(f"`generate_ecograph_raster`:Column '{dist_col}' not found in gdf.")

    dist_series = pd.to_numeric(gdf[dist_col], errors="coerce")
    if dist_series.dropna().empty:
        raise ValueError(
            f"`generate_ecograph_raster`:Column '{dist_col}' has no numeric values to compute a mean resolution."
        )

    if (movement_covariate is None) == (network_metric is None):
        raise ValueError("`generate_ecograph_raster`:Provide exactly one of 'movement_covariate' or 'network_metric'.")

    if output_dir is None or str(output_dir).strip() == "":
        output_dir = os.getcwd()

    output_dir = remove_file_scheme(output_dir)

    if not filename:
        df_hash = hashlib.sha256(pd.util.hash_pandas_object(gdf, index=True).values).hexdigest()
        filename = df_hash[:7]
    if step_length is not None:
        resolution = float(step_length)
    else:
        step_length = float(dist_series.mean())
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
    logger.info(f"Ecograph raster generated at: {raster_path}")
    return raster_path


@task
def retrieve_feature_gdf(
    file_path: Annotated[str, Field(description="Path to the saved Ecograph feature file")],
) -> AnyGeoDataFrame:
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("retrieve_feature_gdf: 'file_path' must be a non-empty string.")

    logger.info(f"Retrieving feature GeoDataFrame from: {file_path}")
    file_path = remove_file_scheme(file_path)
    gdf = get_feature_gdf(file_path)
    return gdf
