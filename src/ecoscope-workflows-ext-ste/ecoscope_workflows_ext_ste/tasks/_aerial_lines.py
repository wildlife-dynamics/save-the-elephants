import numpy as np
import geopandas as gpd
from typing import Literal
from ecoscope.base.utils import hex_to_rgba
from ecoscope_workflows_core.decorators import task
from shapely.geometry import LineString, MultiPolygon
from ecoscope_workflows_core.annotations import AnyGeoDataFrame
from ecoscope_workflows_ext_custom.tasks.transformation._data_cleanup import drop_null_geometry


@task
def validate_polygon_geometry(gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    # Drop null geometries first, before validation
    gdf = drop_null_geometry(gdf)

    # Handle empty GeoDataFrame
    if len(gdf) == 0:
        return gdf

    valid_geoms = {"Polygon", "MultiPolygon"}
    geom_types = set(gdf.geometry.geom_type.unique())
    invalid_types = geom_types - valid_geoms

    if invalid_types:
        raise ValueError(
            f"Invalid geometry types found: {invalid_types}. "
            f"Only Polygon and MultiPolygon are supported for aerial survey lines."
        )

    return gdf


@task
def draw_survey_lines(
    gdf: AnyGeoDataFrame,
    direction: Literal["North South", "East West"] = "North South",
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
    lines_gdf = gpd.overlay(lines_gdf, gdf, how="intersection")

    return lines_gdf


@task
def generate_survey_line_colors(df: AnyGeoDataFrame, value: str) -> AnyGeoDataFrame:
    df["color"] = value
    df["survey_colors"] = df["value"].apply(hex_to_rgba)
