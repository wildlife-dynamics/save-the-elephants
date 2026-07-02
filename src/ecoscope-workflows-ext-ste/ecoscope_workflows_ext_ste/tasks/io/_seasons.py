import pandas as pd
import geopandas as gpd
from pydantic import Field
from datetime import datetime
from wt_registry import register
from typing import Annotated, cast
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

MODIS_START = datetime(2000, 2, 24)


@register()
def compute_seasons_from_ndvi(
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
