import pandas as pd
import geopandas as gpd
from pydantic import Field
from datetime import datetime
from wt_registry import register
from typing import Annotated, cast, Optional
from ecoscope.analysis.seasons import (  # type: ignore[import-untyped]
    seasonal_windows,
    std_ndvi_vals,
    val_cuts,
)
from ecoscope.platform.annotations import (
    AdvancedField,
    AnyDataFrame,
    AnyGeoDataFrame,
)
from ecoscope.platform.tasks.filter._filter import TimeRange
from ecoscope.platform.connections import EarthEngineClient

MODIS_START = datetime(2000, 2, 24)


@register()
def compute_seasons_from_ndvi(
    client: EarthEngineClient,
    roi: Annotated[AnyGeoDataFrame, Field(description="Region of interest.")],
    time_range: Annotated[TimeRange, Field(description="Analysis time range.")],
    img_collection: Annotated[
        str,
        AdvancedField(default="MODIS/061/MCD43A4", description="Earth Engine collection."),
    ] = "MODIS/061/MCD43A4",
    nir_band: Annotated[
        str,
        AdvancedField(default="Nadir_Reflectance_Band2", description="NIR band name."),
    ] = "Nadir_Reflectance_Band2",
    red_band: Annotated[
        str,
        AdvancedField(default="Nadir_Reflectance_Band1", description="Red band name."),
    ] = "Nadir_Reflectance_Band1",
    chunk_count: Annotated[
        int,
        AdvancedField(
            default=5,
            description="Number of chunk boundaries (produces N-1 EE queries). "
            "Increase if Earth Engine times out on long ranges.",
        ),
    ] = 5,
) -> AnyDataFrame:
    """
    Compute seasonal time windows from MODIS NDVI within an ROI.

    Splits the requested time range into chunks, fetches standardized NDVI
    values per chunk via Earth Engine, clusters them into two seasons
    (low/high NDVI), and returns the seasonal windows.

    If the requested `since` predates MODIS coverage (2000-02-24), it is
    clamped forward and a warning is logged.
    """
    modis_start = MODIS_START.replace(tzinfo=time_range.since.tzinfo)
    time_range = time_range.model_copy(update={"since": modis_start})
    merged_roi = cast(gpd.GeoDataFrame, roi).to_crs(4326).union_all()

    chunk_dates = pd.date_range(
        start=time_range.since,
        end=time_range.until,
        periods=chunk_count,
        inclusive="both",
    )
    chunk_strings = [d.isoformat() for d in chunk_dates]

    ndvi_chunks = [
        std_ndvi_vals(
            img_coll=img_collection,
            nir_band=nir_band,
            red_band=red_band,
            aoi=merged_roi,
            start=start,
            end=end,
        )
        for start, end in zip(chunk_strings[:-1], chunk_strings[1:])
    ]
    ndvi_vals = pd.concat(ndvi_chunks)
    cuts = val_cuts(ndvi_vals, 2)
    return cast(AnyDataFrame, seasonal_windows(ndvi_vals, cuts, season_labels=["Dry", "Wet"]))


@register()
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
        if trajectories is None or len(trajectories) == 0:
            raise ValueError("`create_seasonal_labels`: trajectories gdf is empty.")
        if seasons_df is None or len(seasons_df) == 0:
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
                print(f"Found {null_count} NULL values in {col}. These rows will be skipped.")

        seasonal_wins = seasons_df.copy()
        traj_start = trajectories["segment_start"].min()
        traj_end = trajectories["segment_end"].max()

        seasonal_wins = seasonal_wins[
            (seasonal_wins["end"] >= traj_start) & (seasonal_wins["start"] <= traj_end)
        ].reset_index(drop=True)

        print(f"Filtered seasonal windows: {len(seasonal_wins)} periods")
        print(f"Seasonal Windows:\n{seasonal_wins[['start', 'end', 'season']]}")

        if seasonal_wins.empty:
            print("No seasonal windows overlap with trajectory timeframe.")
            trajectories["season"] = None
            return trajectories

        # Validate intervals don't overlap
        seasonal_wins = seasonal_wins.sort_values("start").reset_index(drop=True)
        for i in range(len(seasonal_wins) - 1):
            if seasonal_wins.loc[i, "end"] > seasonal_wins.loc[i + 1, "start"]:
                print(
                    f"Overlapping seasonal windows detected: "
                    f"[{seasonal_wins.loc[i, 'start']} - {seasonal_wins.loc[i, 'end']}] and "
                    f"[{seasonal_wins.loc[i+1, 'start']} - {seasonal_wins.loc[i+1, 'end']}]"
                )

        # Align season window timezone to match trajectory timezone to avoid pd.cut IntervalIndex mismatch
        traj_tz = trajectories["segment_start"].dt.tz
        if traj_tz is not None and seasonal_wins["start"].dt.tz is not None:
            seasonal_wins["start"] = seasonal_wins["start"].dt.tz_convert(traj_tz)
            seasonal_wins["end"] = seasonal_wins["end"].dt.tz_convert(traj_tz)

        season_bins = pd.IntervalIndex(data=seasonal_wins.apply(lambda x: pd.Interval(x["start"], x["end"]), axis=1))
        labels = seasonal_wins["season"].values

        trajectories["season"] = pd.cut(trajectories["segment_start"], bins=season_bins, include_lowest=True).map(
            dict(zip(season_bins, labels))
        )

        null_count = trajectories["season"].isnull().sum()
        if null_count > 0:
            print(f"{null_count} trajectory segments couldn't be assigned to any season")

        trajectories = trajectories.dropna(subset=["season"])
        return trajectories

    except Exception as e:
        print(f"Failed to apply seasonal label to trajectories: {e}")
        trajectories["season"] = None
        return trajectories
