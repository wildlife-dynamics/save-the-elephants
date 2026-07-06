from typing import Annotated

import numpy as np
from wt_registry import register
from ecoscope.platform.annotations import AdvancedField, AnyDataFrame


@register()
def label_by_percentile_threshold(
    gdf: AnyDataFrame,
    column: Annotated[
        str,
        AdvancedField(default="density", description="Column to compute the percentile threshold on."),
    ] = "density",
    pct: Annotated[
        float,
        AdvancedField(default=65, description="Percentile at which to split the two classes."),
    ] = 65,
):
    """
    Split geometries into two classes at the given percentile of value_col.
    pct=65 -> bottom 65% labeled '0-0.65', top 35% labeled '0.65-1'.
    """
    threshold = np.percentile(gdf[column].dropna(), pct)
    frac = pct / 100

    low_label = f"0-{frac:g}"
    high_label = f"{frac:g}-1"

    gdf = gdf.copy()
    gdf["label"] = np.where(gdf[column] >= threshold, high_label, low_label)
    gdf["threshold"] = threshold
    return gdf
