import math
import shapely
import numpy as np
import pandas as pd
import geopandas as gpd
from pydantic import Field
from datetime import datetime
from wt_registry import register
from typing import Annotated, cast, Optional
from shapely.geometry.base import BaseGeometry
from ecoscope.analysis.seasons import (  # type: ignore[import-untyped]
    seasonal_windows,
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

# Pixel size the NDVI mean is taken at; MCD43A4 is a 500 m product.
NDVI_SCALE_METERS = 500
# Longest period fetched in one Earth Engine request, to keep each request quick.
MAX_CHUNK_DAYS = 730
# About 100 m. Trims the vertices of raster-outline ROIs; far below the pixel size.
ROI_SIMPLIFY_DEGREES = 0.001


def _fetch_ndvi(
    aoi: BaseGeometry,
    img_coll: str,
    nir_band: str,
    red_band: str,
    start: str,
    end: str,
    scale: int = NDVI_SCALE_METERS,
) -> pd.DataFrame:
    """Mean NDVI over `aoi` for every image in [start, end), unscaled.

    Used instead of `ecoscope.analysis.seasons.std_ndvi_vals`, which stacks all
    the images of the period into one many-band image and reduces it in a single
    call. With daily MODIS that is thousands of bands, and Earth Engine fails with
    "User memory limit exceeded". Here each image is reduced on its own on the
    server, so memory use does not grow with the length of the period.
    """
    import ee

    geo = ee.Geometry(shapely.geometry.mapping(aoi))

    def mean_ndvi(img):
        mean = (
            img.normalizedDifference([nir_band, red_band])
            .rename("ndvi")
            .reduceRegion(reducer=ee.Reducer.mean(), geometry=geo, scale=scale, bestEffort=True)
            .get("ndvi")
        )
        return ee.Feature(None, {"t": img.get("system:time_start"), "ndvi": mean})

    rows = (
        ee.ImageCollection(img_coll)
        .select([nir_band, red_band])
        .filterDate(start, end)
        .map(mean_ndvi)
        # Images with no valid pixels in the ROI have no mean; drop them.
        .filter(ee.Filter.notNull(["t", "ndvi"]))
        .reduceColumns(ee.Reducer.toList(2), ["t", "ndvi"])
        .get("list")
        .getInfo()
    )
    return pd.DataFrame(
        {
            "img_date": pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
            "NDVI": np.array([r[1] for r in rows], dtype=np.float64),
        }
    )


# Tagged as IO (it calls Earth Engine), so mock-IO test runs use
# compute-seasons-from-ndvi.example-return.parquet instead.
@register(tags=["io"])
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
            description="Minimum number of chunk boundaries (N boundaries produce N-1 Earth Engine "
            "queries). More queries are used automatically when a chunk would cover more than "
            "two years.",
        ),
    ] = 5,
) -> AnyDataFrame:
    """
    Compute seasonal time windows from MODIS NDVI within an ROI.

    Fetches the mean NDVI of every image over the ROI from Earth Engine, in
    chunks of at most two years, clusters the values into two seasons (low/high
    NDVI), and returns the seasonal windows.

    The NDVI history always starts at the beginning of MODIS coverage
    (2000-02-24), whatever `time_range.since` is: the season cut is fitted on
    the long-term record, and only `time_range.until` is used.
    """
    modis_start = MODIS_START.replace(tzinfo=time_range.since.tzinfo)
    time_range = time_range.model_copy(update={"since": modis_start})
    merged_roi = cast(gpd.GeoDataFrame, roi).to_crs(4326).union_all()
    simplified_roi = merged_roi.simplify(ROI_SIMPLIFY_DEGREES)
    if not simplified_roi.is_empty:
        merged_roi = simplified_roi

    span_days = (time_range.until - time_range.since).total_seconds() / 86400
    periods = max(chunk_count, math.ceil(span_days / MAX_CHUNK_DAYS) + 1, 2)
    chunk_dates = pd.date_range(
        start=time_range.since,
        end=time_range.until,
        periods=periods,
        inclusive="both",
    )
    chunk_strings = [d.isoformat() for d in chunk_dates]

    ndvi_chunks = [
        _fetch_ndvi(
            img_coll=img_collection,
            nir_band=nir_band,
            red_band=red_band,
            aoi=merged_roi,
            start=start,
            end=end,
        )
        for start, end in zip(chunk_strings[:-1], chunk_strings[1:])
    ]
    ndvi_vals = pd.concat(ndvi_chunks, ignore_index=True).dropna().sort_values("img_date", ignore_index=True)
    if ndvi_vals.empty:
        raise ValueError(
            f"compute_seasons_from_ndvi: Earth Engine returned no NDVI values for the region of interest "
            f"from {img_collection!r} between {chunk_strings[0]} and {chunk_strings[-1]}."
        )
    # Scale to 0..1 once, over the whole record, so all chunks share one scale.
    ndvi = ndvi_vals["NDVI"].to_numpy()
    ndvi_vals["NDVI"] = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())
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
