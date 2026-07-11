import geopandas as gpd
from pydantic import Field
from wt_registry import register
from typing import Annotated, Literal, cast
from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def spatial_join(
    left_df: Annotated[AnyGeoDataFrame, Field(description="Left GeoDataFrame (e.g. events)")],
    right_df: Annotated[
        AnyGeoDataFrame,
        Field(description="Right GeoDataFrame (e.g. region boundaries)"),
    ],
    how: Annotated[Literal["left", "right", "inner"], Field(description="Join type")] = "left",
    predicate: Annotated[
        Literal["intersects", "contains", "within", "touches", "crosses", "overlaps"],
        Field(description="Spatial predicate for the join"),
    ] = "intersects",
) -> AnyGeoDataFrame:
    """
    Spatially join two GeoDataFrames on a geometric predicate.

    Both frames must share the same CRS — reproject beforehand if not.
    Rows in the preserved frame (per `how`) may be duplicated when one
    geometry matches multiple geometries in the other frame. Overlapping
    non-geometry column names are suffixed with `_left` / `_right`.

    Args:
        left_df: The left GeoDataFrame.
        right_df: The right GeoDataFrame.
        how: Which frame's rows to preserve ("left", "right", or "inner").
        predicate: Spatial relationship to test.

    Returns:
        Joined GeoDataFrame with sjoin's internal index columns stripped.
    """
    if not left_df.crs.equals(right_df.crs):
        raise ValueError(f"CRS mismatch: left={left_df.crs}, right={right_df.crs}.")

    result = gpd.sjoin(
        cast(gpd.GeoDataFrame, left_df),
        cast(gpd.GeoDataFrame, right_df),
        how=how,
        predicate=predicate,
    )
    return cast(
        AnyGeoDataFrame,
        result.drop(columns=[c for c in ("index_left", "index_right") if c in result.columns]),
    )
