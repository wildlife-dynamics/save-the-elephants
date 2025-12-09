import numpy as np
import geopandas as gpd
from pydantic import Field, BaseModel, ConfigDict
from ._path_utils import get_local_geo_path
from typing_extensions import TypeAlias
from typing import Literal, Annotated, Union
from ecoscope_workflows_core.decorators import task
from ._downloader import fetch_and_persist_file
from shapely.geometry import LineString, MultiPolygon
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.transformation._data_cleanup import drop_null_geometry


class DownloadFile(BaseModel):
    model_config = ConfigDict(title="Download from URL")

    url: Annotated[
        str,
        Field(
            title="URL",
            description="URL to download the shapefile from (supports .gpkg, .shp and .geoparquet)",
        ),
    ]


class LocalFile(BaseModel):
    model_config = ConfigDict(title="Use local file")

    file_path: Annotated[
        str,
        Field(
            title="Local file path",
            description="Path to the local shapefile or archive on the filesystem",
        ),
    ]


SelectPath: TypeAlias = Union[DownloadFile, LocalFile]


@task
def get_file_path(
    input_method: SelectPath,
    output_path: str,
) -> str:
    """
    Get file path based on selected input method.
    Returns the path to the (possibly extracted) file/directory ready for use.
    """
    if isinstance(input_method, DownloadFile):
        return fetch_and_persist_file(
            url=input_method.url,
            output_path=output_path,
            unzip=False,
        )
    elif isinstance(input_method, LocalFile):
        return get_local_geo_path(file_path=input_method.file_path)
    else:
        raise ValueError(f"Unsupported input method: {type(input_method)}")


@task
def validate_polygon_geometry(gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    valid_geoms = {"Polygon", "MultiPolygon"}

    geom_types = set(gdf.geometry.geom_type.unique())
    invalid_types = geom_types - valid_geoms

    if invalid_types:
        raise ValueError(
            f"Invalid geometry types found: {invalid_types}. "
            f"Only Polygon and MultiPolygon are supported for aerial survey lines."
        )

    gdf = drop_null_geometry(gdf)
    return gdf


@task
def generate_survey_lines(
    gdf: AnyGeoDataFrame,
    direction: Literal["North South", "East West"] = "NorthSouth",
    spacing: int = 500,
) -> AnyGeoDataFrame:
    """
    Generate parallel survey lines within the boundaries of a polygon GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing polygon boundaries.
        direction (Literal["North South","East West"], optional):Orientation of survey lines. Defaults to "North South".
        spacing (int, optional): Spacing between lines in meters. Defaults to 500.

    Returns:
        GeoDataFrame: GeoDataFrame of clipped survey lines.
    """
    if gdf.crs is None or gdf.crs.to_string() in ["EPSG:4269", "EPSG:4326"]:
        gdf = gdf.to_crs(epsg=3857)

    if any(isinstance(geom, MultiPolygon) for geom in gdf.geometry):
        gdf = gdf.explode(index_parts=False)

    bbox = gdf.total_bounds  # (minx, miny, maxx, maxy)
    lines = []

    if direction == "North South":
        for x in np.arange(bbox[0], bbox[2], spacing):
            lines.append(LineString([(x, bbox[1]), (x, bbox[3])]))
    elif direction == "East West":
        for y in np.arange(bbox[1], bbox[3], spacing):
            lines.append(LineString([(bbox[0], y), (bbox[2], y)]))
    else:
        raise ValueError("Direction must be 'North South' or 'East West'")

    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
    clipped_lines = gpd.overlay(lines_gdf, gdf, how="intersection")

    return clipped_lines
