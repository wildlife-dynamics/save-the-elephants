import pandas as pd
from pydantic import Field
from wt_registry import register
from typing import Annotated, Any, cast
from ecoscope.platform.annotations import AnyDataFrame, AnyGeoDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import (
    AutoScaleGridCellSize,
    CustomGridCellSize,
    calculate_elliptical_time_density,
)


@register()
def calculate_elliptical_time_density_grouped(
    gdf: Annotated[
        AnyGeoDataFrame,
        Field(description="GeoDataFrame to compute home ranges over."),
    ],
    groupby_cols: Annotated[
        list[str],
        Field(
            description="Columns to group by. Each unique combination produces one "
            "home-range result block (e.g., ['season'], ['subject_name'], "
            "['season', 'subject_name'])."
        ),
    ],
    percentiles: Annotated[
        list[float] | None,
        Field(default=None, description="Density percentiles. Defaults to [99.9]."),
    ] = None,
    cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | None,
        Field(
            default=None,
            json_schema_extra={"title": "Grid Cell Size", "ecoscope:advanced": True},
            description="Grid cell strategy. Defaults to AutoScaleGridCellSize().",
        ),
    ] = None,
    drop_null_groups: Annotated[
        bool,
        Field(
            default=True,
            description="If True, drop rows where any groupby column is null before "
            "grouping. If False, null values become their own group.",
        ),
    ] = True,
) -> AnyDataFrame:
    """
    Compute elliptical time-density home ranges per group.

    Trajectories are grouped by `groupby_cols`, and an elliptical time-density
    home range is computed for each unique group combination. Each output row
    block is annotated with the group key values as columns.

    Examples:
        Per season:
        >>> calculate_home_range_by_group(gdf, groupby_cols=["season"])

        Per subject per season:
        >>> calculate_home_range_by_group(
        ...     gdf, groupby_cols=["subject_name", "season"]
        ... )

        Per year:
        >>> calculate_home_range_by_group(gdf, groupby_cols=["year"])
    """
    print(f"[home_range] Computing elliptical home ranges for {len(gdf)} records grouped by {groupby_cols}")
    missing = [c for c in groupby_cols if c not in gdf.columns]
    if missing:
        raise ValueError(f"Missing groupby columns: {missing}. Available: {list(gdf.columns)}")

    percentiles = percentiles or [99.9]
    cell_size = cell_size or AutoScaleGridCellSize()

    if drop_null_groups:
        original_count = len(gdf)
        gdf = gdf.dropna(subset=groupby_cols).copy()
        dropped = original_count - len(gdf)
        if dropped:
            print(f"[home_range]Dropped {dropped} rows with null values")

    if gdf.empty:
        print("[home_range] No records remaining after null filter — returning empty result")
        return cast(AnyDataFrame, pd.DataFrame())

    n_groups = gdf.groupby(groupby_cols).ngroups
    print(f"[home_range] {n_groups} group(s) to process at percentiles {percentiles}")

    results = []
    for group_keys, group in gdf.groupby(groupby_cols, dropna=drop_null_groups):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_label = ", ".join(f"{k}={v}" for k, v in zip(groupby_cols, group_keys))
        print(f"[home_range] Processing group: {group_label} ({len(group)} records)")
        result = calculate_elliptical_time_density(
            group,  # type: ignore[arg-type]
            auto_scale_or_custom_cell_size=cell_size,
            percentiles=percentiles,
        )
        annotation: dict[str, Any] = dict(zip(groupby_cols, group_keys))
        result = result.assign(**annotation)
        results.append(result)

    final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    print(f"[home_range] Completed {len(results)} home range(s) — {len(final)} total output rows")
    return cast(AnyDataFrame, final)
