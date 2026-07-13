import os
from pydantic import Field
from typing import Annotated
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.io._path_utils import remove_file_scheme
from ecoscope.analysis.ecograph import (  # type: ignore[import-untyped]
    get_feature_gdf,
)


@register()
def raster_to_gdf(
    file_path: Annotated[str, Field(description="Path to the saved Ecograph feature file.")],
) -> AnyGeoDataFrame:
    """
    Convert a GeoTIFF feature map into a GeoDataFrame

    Args:
        file_path: Path to the feature file.

    Returns:
        The loaded GeoDataFrame.

    Raises:
        ValueError: If `file_path` is empty.
        FileNotFoundError: If the file doesn't exist.
    """
    print(f"[raster_to_gdf] Loading feature raster from: {file_path}")
    if not file_path or not file_path.strip():
        raise ValueError("'file_path' must be a non-empty string.")

    file_path = remove_file_scheme(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")

    result = get_feature_gdf(file_path)
    print(f"[raster_to_gdf] Loaded {len(result)} rows with columns: {list(result.columns)}")
    return result
