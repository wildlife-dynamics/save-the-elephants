import numpy as np
import geopandas as gpd
from pydantic import Field
from wt_registry import register
from typing import Annotated, Literal, cast
from ecoscope.platform.annotations import AnyGeoDataFrame


@register()
def spatial_tag(
    df: Annotated[
        AnyGeoDataFrame,
        Field(description="The GeoDataFrame to tag. Each row will receive a label."),
    ],
    reference_gdf: Annotated[
        AnyGeoDataFrame,
        Field(
            description="The reference layer containing geometries to test against "
            "(e.g., protected areas, ecoregions, admin boundaries)."
        ),
    ],
    output_column: Annotated[
        str,
        Field(description="Name of the column to add to `df` with the tag result."),
    ],
    matched_label: Annotated[
        str,
        Field(
            default="Inside",
            description="Label for rows whose geometry matches the predicate against "
            "any geometry in `reference_gdf`.",
        ),
    ] = "Inside",
    unmatched_label: Annotated[
        str,
        Field(
            default="Outside",
            description="Label for rows that don't match any geometry in `reference_gdf`.",
        ),
    ] = "Outside",
    predicate: Annotated[
        Literal["intersects", "within", "contains"],
        Field(
            default="intersects",
            description="Spatial predicate. 'intersects' matches geometries that touch "
            "at all; 'within' requires full containment in a reference geometry; "
            "'contains' tags rows whose geometry fully contains a reference geometry.",
        ),
    ] = "intersects",
) -> AnyGeoDataFrame:
    """
    Tag each row of `df` based on its spatial relationship with `reference_gdf`.

    If `reference_gdf` is in a different CRS than `df`, it is reprojected to match.
    Both inputs must have a CRS set. The output preserves the columns, row count,
    and CRS of `df` exactly.

    Examples:
        Tag points by whether they fall inside protected areas:
        >>> spatial_tag(
        ...     df=sightings_gdf,
        ...     reference_gdf=protected_areas_gdf,
        ...     output_column="protection_status",
        ...     matched_label="Protected",
        ...     unmatched_label="Unprotected",
        ... )

        Tag tracks by ecoregion membership:
        >>> spatial_tag(
        ...     df=tracks_gdf,
        ...     reference_gdf=ecoregions_gdf,
        ...     output_column="in_target_ecoregion",
        ...     matched_label="Yes",
        ...     unmatched_label="No",
        ... )
    """
    if df.crs is None:
        raise ValueError("`df` must have a CRS set.")
    if reference_gdf.crs is None:
        raise ValueError("`reference_gdf` must have a CRS set.")

    if not reference_gdf.crs.equals(df.crs):
        print(
            "Reprojecting reference_gdf from %s to %s to match df.",
            reference_gdf.crs,
            df.crs,
        )
        reference_gdf = cast(AnyGeoDataFrame, cast(gpd.GeoDataFrame, reference_gdf).to_crs(df.crs))

    out = df.copy()

    if reference_gdf.empty:
        print(
            "reference_gdf is empty; all rows will be tagged as '%s'.",
            unmatched_label,
        )
        out[output_column] = unmatched_label
        return cast(AnyGeoDataFrame, out)

    reference_union = reference_gdf.geometry.union_all()
    predicate_map = {
        "intersects": out.geometry.intersects,
        "within": out.geometry.within,
        "contains": out.geometry.contains,
    }
    out[output_column] = np.where(predicate_map[predicate](reference_union), matched_label, unmatched_label)
    return cast(AnyGeoDataFrame, out)
