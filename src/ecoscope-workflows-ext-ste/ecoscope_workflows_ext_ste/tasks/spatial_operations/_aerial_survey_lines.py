import logging
import numpy as np
import geopandas as gpd
from wt_registry import register
from typing import Literal, Optional, cast
from shapely.geometry import LineString, MultiPolygon
from ecoscope.platform.annotations import (  # type: ignore[import-untyped]
    AnyGeoDataFrame,
)

logger = logging.getLogger(__name__)


@register()
def ensure_polygon_type(gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:  # assert_polygon_type
    """
    Assert all remaining geometries are
    Polygon or MultiPolygon.
    """
    geom_types = set(gdf.geometry.geom_type.unique())
    print(f"[ensure_polygon_type] Validating {len(gdf)} geometries — types found: {geom_types}")
    invalid = geom_types - {"Polygon", "MultiPolygon"}
    if invalid:
        raise ValueError(f"Invalid geometry types: {invalid}. " f"Only Polygon and MultiPolygon are supported.")
    print(f"[ensure_polygon_type] All {len(gdf)} geometries are valid polygons.")
    return gdf


@register()
def create_survey_transects(  # generate_survey_transects
    gdf: AnyGeoDataFrame,
    direction: Literal["North South", "East West"] = "North South",
    spacing: int = 300,
    planar_crs: Optional[str] = None,
) -> AnyGeoDataFrame:
    """
    Generate parallel survey lines within polygon boundaries, returned in
    the input CRS.

    Spacing is measured in meters in `planar_crs`. By default, the function
    estimates an appropriate UTM CRS from the input geometry. For analyses
    that need a specific projection (e.g., a regional equal-area), pass
    `planar_crs` explicitly.

    Args:
        gdf: GeoDataFrame containing polygon boundaries with a CRS set.
        direction: Orientation of survey lines.
        spacing: Spacing between lines in meters (in `planar_crs`).
        planar_crs: Optional CRS string used for the spacing calculation.
            Defaults to the auto-estimated UTM zone for `gdf`.

    Returns:
        GeoDataFrame of clipped survey lines, in the input CRS.

    Raises:
        ValueError: If `gdf` is empty or has no CRS set.
    """
    _gdf: gpd.GeoDataFrame = cast(gpd.GeoDataFrame, gdf)
    original_crs = _gdf.crs
    if original_crs is None:
        raise ValueError("Input GeoDataFrame must have a CRS set.")
    target_crs = planar_crs or _gdf.estimate_utm_crs()
    projected = _gdf.to_crs(target_crs)

    if any(isinstance(geom, MultiPolygon) for geom in projected.geometry):
        projected = projected.explode(index_parts=False)

    minx, miny, maxx, maxy = projected.total_bounds
    lines = []

    if direction == "North South":
        for x in np.arange(minx, maxx, spacing):
            lines.append(LineString([(x, miny), (x, maxy)]))
    elif direction == "East West":
        for y in np.arange(miny, maxy, spacing):
            lines.append(LineString([(minx, y), (maxx, y)]))
    else:
        raise ValueError("direction must be 'North South' or 'East West'.")

    print(f"[create_survey_transects] Generated {len(lines)} candidate lines — clipping to polygon boundaries...")
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=projected.crs)
    lines_gdf = gpd.overlay(lines_gdf, projected, how="intersection")

    result = cast(AnyGeoDataFrame, lines_gdf.to_crs(original_crs.to_wkt()))
    print(f"[create_survey_transects] {len(result)} transects after clipping")
    return result
