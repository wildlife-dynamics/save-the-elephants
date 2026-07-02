from typing import cast
import geopandas as gpd
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def compute_minimum_convex_polygon(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",
) -> AnyGeoDataFrame:
    """
    Compute the Minimum Convex Polygon (MCP) of input geometries and its area.

    The MCP is computed in the given equal-area projection (`planar_crs`) so
    that the reported area is meaningful, then projected back to the input
    CRS for the output geometry. Non-point geometries are accepted; the
    convex hull is computed over all of their vertices.

    Args:
        gdf: Input GeoDataFrame with a CRS set.
        planar_crs: Equal-area projection used for the area calculation.
            Defaults to Africa Albers Equal Area.

    Returns:
        Single-row GeoDataFrame with the MCP polygon (in the input CRS),
        plus `area_m2` and `area_km2` columns.

    Raises:
        ValueError: If `gdf` is empty, has no geometry column, has no CRS,
            contains no valid geometries, or has fewer than 3 unique vertices
            across all geometries.
    """
    print(f"[mcp] Computing minimum convex polygon for {len(gdf)} input geometries (planar CRS: {planar_crs})")
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set. Please assign a CRS before computing the MCP.")

    original_crs = gdf.crs
    valid = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()]
    if valid.empty:
        raise ValueError("Input GeoDataFrame is empty or contains no valid geometries.")

    projected = valid.to_crs(planar_crs)
    unioned = projected.geometry.union_all()

    # Count unique vertices across all geometries; MCP needs at least 3
    # non-collinear points to form a polygon.
    coords = set()
    for geom in getattr(unioned, "geoms", [unioned]):
        coords.update(geom.exterior.coords if geom.geom_type == "Polygon" else geom.coords)
    if len(coords) < 3:
        raise ValueError(f"MCP requires at least 3 unique points; got {len(coords)}.")

    convex_hull = unioned.convex_hull
    if convex_hull.geom_type != "Polygon":
        raise ValueError("MCP requires at least 3 non-collinear points; " f"got a {convex_hull.geom_type}.")

    area_m2 = float(convex_hull.area)
    hull_in_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {
            "area_m2": [area_m2],
            "area_km2": [area_m2 / 1e6],
        },
        geometry=[hull_in_original_crs],
        crs=original_crs,
    )
    print(f"[mcp] MCP area: {area_m2 / 1e6:.2f} km² ({area_m2:,.0f} m²) from {len(coords)} unique vertices")
    return cast(AnyGeoDataFrame, result_gdf)
