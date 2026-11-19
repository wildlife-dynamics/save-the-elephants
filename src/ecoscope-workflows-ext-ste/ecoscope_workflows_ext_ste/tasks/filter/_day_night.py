import numpy as np
import geopandas as gpd
from wt_registry import register
from ecoscope.platform.annotations import AnyGeoDataFrame
from .._log import count, log


@register()
def aggregate_day_night_fixes(gdf: AnyGeoDataFrame) -> AnyGeoDataFrame:
    """
    Count night/day records per geometry and flag which one dominates.
    Returns a GeoDataFrame with columns: geometry, day_count, night_count, dominant.
    """
    counts = gdf.groupby("geometry")["is_night"].value_counts().unstack(fill_value=0).reset_index()

    # Ensure both columns exist even if data is all-night or all-day
    for val in [False, True]:
        if val not in counts.columns:
            counts[val] = 0

    counts = counts.rename(columns={False: "day_count_fixes", True: "night_count_fixes"})
    counts["dominant"] = np.where(counts["night_count_fixes"] > counts["day_count_fixes"], "Night", "Day")

    night = int((counts["dominant"] == "Night").sum())
    log(
        "tracks",
        f"day/night: {count(len(gdf), 'fix', 'fixes')} at {count(len(counts), 'location')}, "
        f"{night:,} mostly at night and {len(counts) - night:,} mostly by day",
    )
    return gpd.GeoDataFrame(counts, geometry="geometry", crs=gdf.crs)
