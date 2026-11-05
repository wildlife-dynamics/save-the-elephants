import os
import hashlib
import pandas as pd
from pydantic import Field
from wt_registry import register
from typing import Annotated, Literal, Optional
from rasterio.io import MemoryFile  # type: ignore[import-untyped]
from ecoscope.analysis.ecograph import (  # type: ignore[import-untyped]
    Ecograph,
    get_feature_gdf,
)
from ecoscope.trajectory import Trajectory  # type: ignore[import-untyped]
from ecoscope.platform.annotations import AdvancedField, AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme


@register()
def generate_ecograph_raster(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data.")],
    movement_covariate: Annotated[
        Optional[
            Literal[
                "dot_product",
                "step_length",
                "speed",
                "sin_time",
                "cos_time",
                "tortuosity_1",
                "tortuosity_2",
            ]
        ],
        AdvancedField(
            default=None,
            description="Movement covariate to rasterize. Mutually exclusive with `network_metric`.",
        ),
    ] = None,
    network_metric: Annotated[
        Optional[Literal["weight", "betweenness", "degree", "collective_influence"]],
        AdvancedField(
            default=None,
            description="Network metric to rasterize. Mutually exclusive with `movement_covariate`.",
        ),
    ] = None,
    dist_col: Annotated[
        str,
        AdvancedField(default="dist_meters", description="Column name for step distance."),
    ] = "dist_meters",
    output_dir: Annotated[
        Optional[str],
        AdvancedField(default=None, description="Directory to save the output raster. Defaults to CWD."),
    ] = None,
    filename: Annotated[
        Optional[str],
        AdvancedField(
            default=None,
            description="Filename for the output GeoTIFF (no extension). " "If None, a hash of the input data is used.",
            exclude=True,
        ),
    ] = None,
    resolution: Annotated[
        Optional[float],
        AdvancedField(
            default=None, description="Raster resolution. If None, uses `step_length` or the mean of `dist_col`."
        ),
    ] = None,
    radius: Annotated[
        int,
        AdvancedField(default=2, description="Radius for kernel smoothing."),
    ] = 2,
    cutoff: Annotated[
        Optional[float],
        AdvancedField(default=None, description="Cutoff distance for kernel."),
    ] = None,
    tortuosity_length: Annotated[
        int,
        AdvancedField(default=3, description="Length scale for tortuosity smoothing."),
    ] = 3,
    interpolation: Annotated[
        Literal["mean", "min", "max", "median"],
        AdvancedField(default="mean", description="Aggregation method when multiple values fall in a cell."),
    ] = "mean",
    step_length: Annotated[
        Optional[int],
        AdvancedField(
            default=2000,
            gt=500,
            lt=100000,
            description="Mean step length, used as the resolution if `resolution` is None.",
        ),
    ] = 2000,
) -> str:
    """
    Build an Ecograph from trajectory data and write a raster of one
    movement covariate or network metric to GeoTIFF.

    Resolution precedence: explicit `resolution` > `step_length` > mean of
    `dist_col`. Provide exactly one of `movement_covariate` or
    `network_metric`.

    Args:
        gdf: Non-empty GeoDataFrame of trajectory segments.
        dist_col: Column with step distances (numeric).
        output_dir: Where to write the GeoTIFF. Created if missing.
            Defaults to current working directory.
        filename: Output filename without extension. Defaults to a 12-char
            hash of the input data.
        resolution: Raster resolution in source units; takes priority over
            `step_length` and the dist_col mean.
        radius: Kernel-smoothing radius.
        cutoff: Kernel cutoff distance.
        tortuosity_length: Length scale for tortuosity smoothing.
        interpolation: Aggregation method when multiple values fall in a cell.
        step_length: Used as resolution if `resolution` is None.
        movement_covariate: Movement covariate to rasterize. Mutually
            exclusive with `network_metric`.
        network_metric: Network metric to rasterize. Mutually exclusive
            with `movement_covariate`.

    Returns:
        Absolute path to the written GeoTIFF.

    Raises:
        ValueError: For empty inputs, missing columns, non-positive
            resolution, or violation of the mutual-exclusion rule.
    """
    ecograph, covariate = _build_ecograph(
        gdf,
        movement_covariate=movement_covariate,
        network_metric=network_metric,
        dist_col=dist_col,
        resolution=resolution,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length,
        step_length=step_length,
    )

    output_dir = remove_file_scheme((output_dir or os.getcwd()).strip())
    os.makedirs(output_dir, exist_ok=True)

    if not filename:
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(gdf.drop(columns="geometry"), index=True).values  # type: ignore[arg-type]
        ).hexdigest()
        filename = df_hash[:12]

    raster_path = os.path.join(output_dir, f"{filename}.tif")
    ecograph.to_geotiff(covariate, raster_path, interpolation=interpolation)
    print(f"[generate_ecograph_raster] Raster saved: {raster_path}")
    return raster_path


def _build_ecograph(
    gdf: AnyGeoDataFrame,
    movement_covariate: Optional[str],
    network_metric: Optional[str],
    dist_col: str,
    resolution: Optional[float],
    radius: int,
    cutoff: Optional[float],
    tortuosity_length: int,
    step_length: Optional[int],
) -> tuple[Ecograph, str]:
    """Validate the inputs and build the Ecograph.

    Returns the Ecograph and the name of the feature to rasterize.
    """
    provided = [x for x in (movement_covariate, network_metric) if x is not None]
    if len(provided) != 1:
        raise ValueError("Provide exactly one of 'movement_covariate' or 'network_metric'.")

    dist_numeric = pd.to_numeric(gdf[dist_col], errors="coerce").dropna()
    if dist_numeric.empty:
        raise ValueError(f"Column '{dist_col}' has no numeric values to compute resolution.")

    if resolution is not None:
        res = float(resolution)
    elif step_length is not None:
        res = float(step_length)
    else:
        res = float(dist_numeric.mean())

    if res <= 0:
        raise ValueError(f"Resolution must be > 0, got {res}.")

    ecograph = Ecograph(
        Trajectory(gdf),
        resolution=res,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length,
    )
    return ecograph, provided[0]


@register()
def generate_ecograph_gdf(
    gdf: Annotated[AnyGeoDataFrame, Field(description="GeoDataFrame with trajectory data.")],
    movement_covariate: Annotated[
        Optional[
            Literal[
                "dot_product",
                "step_length",
                "speed",
                "sin_time",
                "cos_time",
                "tortuosity_1",
                "tortuosity_2",
            ]
        ],
        AdvancedField(
            default=None,
            description="Movement covariate to rasterize. Mutually exclusive with `network_metric`.",
        ),
    ] = None,
    network_metric: Annotated[
        Optional[Literal["weight", "betweenness", "degree", "collective_influence"]],
        AdvancedField(
            default=None,
            description="Network metric to rasterize. Mutually exclusive with `movement_covariate`.",
        ),
    ] = None,
    dist_col: Annotated[
        str,
        AdvancedField(default="dist_meters", description="Column name for step distance."),
    ] = "dist_meters",
    resolution: Annotated[
        Optional[float],
        AdvancedField(
            default=None, description="Raster resolution. If None, uses `step_length` or the mean of `dist_col`."
        ),
    ] = None,
    radius: Annotated[
        int,
        AdvancedField(default=2, description="Radius for kernel smoothing."),
    ] = 2,
    cutoff: Annotated[
        Optional[float],
        AdvancedField(default=None, description="Cutoff distance for kernel."),
    ] = None,
    tortuosity_length: Annotated[
        int,
        AdvancedField(default=3, description="Length scale for tortuosity smoothing."),
    ] = 3,
    interpolation: Annotated[
        Literal["mean", "min", "max", "median"],
        AdvancedField(default="mean", description="Aggregation method when multiple values fall in a cell."),
    ] = "mean",
    step_length: Annotated[
        Optional[int],
        AdvancedField(
            default=2000,
            gt=500,
            lt=100000,
            description="Mean step length, used as the resolution if `resolution` is None.",
        ),
    ] = 2000,
) -> AnyGeoDataFrame:
    """
    Build an Ecograph from trajectory data and return one movement covariate
    or network metric as a GeoDataFrame of raster-cell polygons.

    Same inputs as `generate_ecograph_raster`, but the raster is written to
    an in-memory GDAL file (/vsimem/) and read straight back with
    `get_feature_gdf`, so nothing is written to local disk or cloud storage.
    The result matches `raster_to_gdf(generate_ecograph_raster(...))`: a
    `value` column and polygon geometry in the trajectory's UTM CRS.

    Raises:
        ValueError: For empty inputs, missing columns, non-positive
            resolution, or violation of the mutual-exclusion rule.
    """
    ecograph, covariate = _build_ecograph(
        gdf,
        movement_covariate=movement_covariate,
        network_metric=network_metric,
        dist_col=dist_col,
        resolution=resolution,
        radius=radius,
        cutoff=cutoff,
        tortuosity_length=tortuosity_length,
        step_length=step_length,
    )
    with MemoryFile() as mem:
        ecograph.to_geotiff(covariate, mem.name, interpolation=interpolation)
        result = get_feature_gdf(mem.name)
    print(f"[generate_ecograph_gdf] Built {len(result)} raster features for '{covariate}'.")
    return result
