import geopandas as gpd
from pydantic import Field
from typing import Annotated,Dict
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame, AdvancedField

# this is useful when you want to label different parts of a gdf separately
@task
def split_gdf_by_column(
    gdf: Annotated[AnyGeoDataFrame, Field(description="The GeoDataFrame to split")],
    column: Annotated[str, Field(description="Column name to split GeoDataFrame by")],
) -> Dict[str, AnyGeoDataFrame]:
    """
    Splits a GeoDataFrame into a dictionary of GeoDataFrames based on unique values in the specified column.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`split_gdf_by_column`:gdf is empty.")

    if column not in gdf.columns:
        raise ValueError(f"`split_gdf_by_column`:Column '{column}' not found in GeoDataFrame.")

    grouped = {str(k): v for k, v in gdf.groupby(column)}
    return grouped

@task
def generate_mcp_gdf(
    gdf: AnyGeoDataFrame,
    planar_crs: str = "ESRI:102022",  # Africa Albers Equal Area
) -> AnyGeoDataFrame:
    """
    Create a Minimum Convex Polygon (MCP) from input point geometries and compute its area.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`generate_mcp_gdf`:gdf is empty.")
    if gdf.geometry is None:
        raise ValueError("`generate_mcp_gdf`:gdf has no 'geometry' column.")
    if gdf.crs is None:
        raise ValueError("`generate_mcp_gdf`:gdf must have a CRS set (e.g., EPSG:4326).")

    original_crs = gdf.crs
    valid_points_gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    if valid_points_gdf.empty:
        raise ValueError("`generate_mcp_gdf`:No valid geometries in gdf.")
    
    projected_gdf = valid_points_gdf.to_crs(planar_crs)
    if not all(projected_gdf.geometry.geom_type.isin(["Point"])):
        projected_gdf.geometry = projected_gdf.geometry.centroid

    convex_hull = projected_gdf.geometry.unary_union.convex_hull

    area_sq_meters = float(convex_hull.area)
    area_sq_km = area_sq_meters / 1_000_000.0
    convex_hull_original_crs = gpd.GeoSeries([convex_hull], crs=planar_crs).to_crs(original_crs).iloc[0]

    result_gdf = gpd.GeoDataFrame(
        {
            "area_m2": [area_sq_meters], 
            "area_km2": [area_sq_km], 
            "mcp": "mcp"
        },
        geometry=[convex_hull_original_crs],
        crs=original_crs,
    )
    return result_gdf