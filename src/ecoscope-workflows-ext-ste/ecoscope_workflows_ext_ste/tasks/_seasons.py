import logging
import pandas as pd
from pydantic import Field
from typing import Optional, Annotated
from pydantic.json_schema import SkipJsonSchema
from ecoscope_workflows_core.decorators import task
from ecoscope_workflows_core.annotations import AnyGeoDataFrame, AnyDataFrame
from ecoscope_workflows_ext_ecoscope.tasks.analysis._time_density import (
    AutoScaleGridCellSize,
    CustomGridCellSize,
    calculate_elliptical_time_density,
)

logger = logging.getLogger(__name__)


@task
def create_seasonal_labels(trajectories: AnyGeoDataFrame, seasons_df: AnyDataFrame) -> Optional[AnyGeoDataFrame]:
    """
    Annotates trajectory segments with seasonal labels (wet/dry) based on NDVI-derived windows.
    Applies to the entire trajectory without grouping.

    Args:
        trajectories: GeoDataFrame containing trajectory segments with 'segment_start' and 'segment_end'.
        seasons_df: DataFrame containing seasonal windows with 'start', 'end', and 'season' columns.

    Returns:
        GeoDataFrame with a new 'season' column containing the assigned seasonal label.
        Rows that could not be assigned a season are dropped.
        Returns None if an error occurs.
    """
    try:
        # Validate input DataFrames are not empty
        if trajectories is None or trajectories.empty:
            raise ValueError("`create_seasonal_labels`: trajectories gdf is empty.")
        if seasons_df is None or seasons_df.empty:
            raise ValueError("`create_seasonal_labels`: seasons_df is empty.")

        # Validate required columns in trajectories
        required_traj_cols = ["segment_start", "segment_end"]
        missing_traj_cols = [col for col in required_traj_cols if col not in trajectories.columns]
        if missing_traj_cols:
            raise ValueError(
                f"`create_seasonal_labels`: trajectories is missing required columns: {missing_traj_cols}. "
                f"Available columns: {list(trajectories.columns)}"
            )

        # Validate required columns in seasons_df
        required_season_cols = ["start", "end", "season"]
        missing_season_cols = [col for col in required_season_cols if col not in seasons_df.columns]
        if missing_season_cols:
            raise ValueError(
                f"`create_seasonal_labels`: seasons_df is missing required columns: {missing_season_cols}. "
                f"Available columns: {list(seasons_df.columns)}"
            )

        # Validate datetime types
        for col in ["segment_start", "segment_end"]:
            if not pd.api.types.is_datetime64_any_dtype(trajectories[col]):
                raise TypeError(f"`{col}` must be datetime type, got {trajectories[col].dtype}")
        for col in ["start", "end"]:
            if not pd.api.types.is_datetime64_any_dtype(seasons_df[col]):
                raise TypeError(f"`{col}` must be datetime type, got {seasons_df[col].dtype}")

        # Warn for NULL values in critical columns
        for col in ["segment_start", "segment_end"]:
            null_count = trajectories[col].isnull().sum()
            if null_count > 0:
                logger.warning(f"Found {null_count} NULL values in {col}. These rows will be skipped.")

        seasonal_wins = seasons_df.copy()
        traj_start = trajectories["segment_start"].min()
        traj_end = trajectories["segment_end"].max()

        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        logger.info(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        logger.info(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            logger.error("No seasonal windows overlap with trajectory timeframe.")
            trajectories["season"] = None
            return trajectories

        # Validate intervals don't overlap
        seasonal_wins = seasonal_wins.sort_values("start").reset_index(drop=True)
        for i in range(len(seasonal_wins) - 1):
            if seasonal_wins.loc[i, "end"] > seasonal_wins.loc[i + 1, "start"]:
                logger.warning(
                    f"Overlapping seasonal windows detected: "
                    f"[{seasonal_wins.loc[i, 'start']} - {seasonal_wins.loc[i, 'end']}] and "
                    f"[{seasonal_wins.loc[i+1, 'start']} - {seasonal_wins.loc[i+1, 'end']}]"
                )

        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        labels = seasonal_wins["season"].values

        trajectories["season"] = pd.cut(trajectories["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )

        null_count = trajectories["season"].isnull().sum()
        if null_count > 0:
            logger.warning(f"{null_count} trajectory segments couldn't be assigned to any season")

        trajectories = trajectories.dropna(subset=["season"])
        return trajectories

    except Exception as e:
        logger.error(f"Failed to apply seasonal label to trajectories: {e}")
        return None


@task
def calculate_seasonal_home_ranger(
    gdf: AnyGeoDataFrame,
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'season'])",
            json_schema_extra={"default": ["groupby_col", "season"]},
        ),
    ] = None,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = [99.9],
    auto_scale_or_custom_cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
        Field(
            json_schema_extra={
                "title": "Auto Scale Or Custom Grid Cell Size",
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom": "Auto-scale"},
            },
        ),
    ] = None,
) -> AnyDataFrame:
    if gdf is None or gdf.empty:
        raise ValueError("`calculate_seasonal_home_range`:gdf is empty.")

    if groupby_cols is None:
        groupby_cols = ["groupby_col", "season"]

    if "season" not in gdf.columns:
        raise ValueError("`calculate_seasonal_home_range`: gdf must have a 'season' column.")

    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    gdf = gdf[gdf["season"].notna()].copy()
    try:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df,
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            )
        )
    except TypeError:
        season_etd = gdf.groupby(groupby_cols).apply(
            lambda df: calculate_elliptical_time_density(
                df,
                auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
                percentiles=percentiles,
            ),
            include_groups=False,
        )
    # Reset index properly
    if isinstance(season_etd.index, pd.MultiIndex):
        season_etd = season_etd.reset_index()
    return season_etd


@task
def calculate_seasonal_home_range(
    gdf: AnyGeoDataFrame,
    groupby_cols: Annotated[
        list[str],
        Field(
            description="List of column names to group by (e.g., ['groupby_col', 'season'])",
            json_schema_extra={"default": ["groupby_col", "season"]},
        ),
    ] = None,
    percentiles: Annotated[
        list[float] | SkipJsonSchema[None],
        Field(default=[25.0, 50.0, 75.0, 90.0, 95.0, 99.9]),
    ] = [99.9],
    auto_scale_or_custom_cell_size: Annotated[
        AutoScaleGridCellSize | CustomGridCellSize | SkipJsonSchema[None],
        Field(
            json_schema_extra={
                "title": "Auto Scale Or Custom Grid Cell Size",
                "ecoscope:advanced": True,
                "default": {"auto_scale_or_custom": "Auto-scale"},
            },
        ),
    ] = None,
) -> AnyDataFrame:
    """
    Calculate seasonal home ranges using elliptical time density.
    Hardened against MultiIndex + pandas.isna() crashes.
    """
    if gdf is None or gdf.empty:
        raise ValueError("`calculate_seasonal_home_range`: gdf is empty or None.")

    if groupby_cols is None:
        groupby_cols = ["groupby_col", "season"]

    if "season" not in gdf.columns:
        raise ValueError("`calculate_seasonal_home_range`: gdf must contain a 'season' column.")

    if auto_scale_or_custom_cell_size is None:
        auto_scale_or_custom_cell_size = AutoScaleGridCellSize()

    # Early cleaning - remove rows without season
    gdf = gdf[gdf["season"].notna()].copy()

    # Helper function to compute ETD for a single group
    def compute_etd(group_df):
        # Critical safety: reset index BEFORE calling the potentially dangerous function
        group_df = group_df.reset_index(drop=True)

        return calculate_elliptical_time_density(
            group_df,
            auto_scale_or_custom_cell_size=auto_scale_or_custom_cell_size,
            percentiles=percentiles,
        )

    try:
        # Preferred modern pattern (pandas 2.0+)
        season_etd = (
            gdf.groupby(groupby_cols, group_keys=False).apply(compute_etd).reset_index(drop=True)  # Final safety net
        )
    except TypeError:
        # Fallback for older pandas or strict include_groups behavior
        season_etd = (
            gdf.groupby(groupby_cols, group_keys=False).apply(compute_etd, include_groups=False).reset_index(drop=True)
        )

    # Extra defensive layer (rarely needed after the above, but cheap insurance)
    if isinstance(season_etd.index, pd.MultiIndex):
        season_etd = season_etd.reset_index(drop=True)

    # Optional: ensure no leftover categorical index issues
    if season_etd.index.name is not None:
        season_etd = season_etd.reset_index(drop=True)

    return season_etd
