import numpy as np
import geopandas as gpd
from typing import Literal
from shapely.geometry import LineString, MultiPolygon
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame

@task
def generate_survey_lines(
    gdf: AnyGeoDataFrame,
    direction: Literal["N-S", "E-W"] = "N-S",
    spacing: int = 500,
) -> AnyGeoDataFrame:
    """
    Generate parallel survey lines within the boundaries of a polygon GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame containing polygon boundaries.
        direction (Literal["N-S", "E-W"], optional): Orientation of survey lines. Defaults to "N-S".
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

    if direction == "N-S":
        for x in np.arange(bbox[0], bbox[2], spacing):
            lines.append(LineString([(x, bbox[1]), (x, bbox[3])]))
    elif direction == "E-W":
        for y in np.arange(bbox[1], bbox[3], spacing):
            lines.append(LineString([(bbox[0], y), (bbox[2], y)]))
    else:
        raise ValueError("Direction must be 'N-S' or 'E-W'")

    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
    clipped_lines = gpd.overlay(lines_gdf, gdf, how="intersection")

    return clipped_lines
