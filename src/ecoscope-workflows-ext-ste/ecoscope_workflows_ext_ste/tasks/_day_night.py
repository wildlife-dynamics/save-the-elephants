import geopandas as gpd
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame


@task
def get_day_night_dominance(points_gdf: AnyGeoDataFrame, grid_gdf: AnyGeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially joins points to a meshgrid and assigns each cell a majority is_night label.

    Args:
        points_gdf: GeoDataFrame with point geometries and an `is_night` boolean column.
        grid_gdf:   GeoDataFrame with meshgrid polygon geometries.

    Returns:
        GeoDataFrame of grid cells that contain at least one point,
        with an `is_night_majority` column valued 'Night' or 'Day'.
    """

    def majority_is_night(series):
        return series.sum() > (len(series) / 2)

    points_projected = points_gdf.to_crs(grid_gdf.crs)

    joined = gpd.sjoin(points_projected[["geometry", "is_night"]], grid_gdf, how="inner", predicate="within")

    majority = (
        joined.groupby("index_right")["is_night"]
        .agg(majority_is_night)
        .reset_index()
        .rename(columns={"is_night": "is_night_majority", "index_right": "grid_index"})
    )

    majority["is_night_majority"] = majority["is_night_majority"].map({True: "Night", False: "Day"})

    grid_result = grid_gdf.merge(majority, left_index=True, right_on="grid_index", how="left")
    grid_result["is_night_majority"] = grid_result["is_night_majority"].fillna("undefined")
    grid_result = grid_result[grid_result["is_night_majority"] != "undefined"].reset_index(drop=True)

    return grid_result


@task
def get_grid_night_fixes(
    points_gdf: AnyGeoDataFrame, grid_gdf: AnyGeoDataFrame, threshold: float = 0.5
) -> AnyGeoDataFrame:
    """
    Filters to night fixes only, counts them per grid cell, then labels each cell
    based on a threshold proportion of the max night point count.

    Args:
        points_gdf: GeoDataFrame with point geometries and an `is_night` boolean column.
        grid_gdf:   GeoDataFrame with meshgrid polygon geometries.
        threshold:  Float between 0 and 1. Fraction of the max night point count.
                    Cells with count >= threshold * max_count → '{threshold} - 1'
                    Cells with count <  threshold * max_count → '0 - {threshold}'

    Returns:
        GeoDataFrame of grid cells containing at least one night fix,
        with `night_point_count` and `night_activity` columns.
    """
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold must be between 0 and 1.")

    night_gdf = points_gdf[points_gdf["is_night"]].to_crs(grid_gdf.crs)

    joined = gpd.sjoin(night_gdf[["geometry", "is_night"]], grid_gdf, how="inner", predicate="within")

    counts = (
        joined.groupby("index_right")["is_night"]
        .count()
        .reset_index()
        .rename(columns={"is_night": "night_point_count", "index_right": "grid_index"})
    )

    max_count = counts["night_point_count"].mean()
    cutoff = threshold * max_count
    counts["night_activity"] = counts["night_point_count"].apply(
        lambda c: f"{threshold} - 1" if c >= cutoff else f"0 - {threshold}"
    )

    grid_result = (
        grid_gdf.merge(counts, left_index=True, right_on="grid_index", how="left")
        .dropna(subset=["night_activity"])
        .reset_index(drop=True)
    )

    return grid_result
